import os
import argparse
from datetime import datetime
from evaluation.metrics import iou, dice, pixel_accuracy
from evaluation.robustness import gaussian_noise_transform, gaussian_blur_transform, contrast_modify_transform, brightness_adjust_transform, apply_occlusion, s_and_p_noise_transform
from models.unet import UNet
from models.autoencoder import Autoencoder
from models.autoencoder_segmentation import AutoEncoderSegmentation
import torch
import torch.backends.cudnn as cudnn
from util import logger
from util.constants import CONSTANTS
from util.data_loader import get_seg_data_loaders, get_data_loaders
from util.model_handler import load_selected_model
import json

log = logger.setup_logger()


class EvaluationRunner:
    """
    Evaluation Runner class for evaluating the model against
    several metrics and robustness measures.
    """

    def __init__(self, model_name, model, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = model.to(self.device)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.results_file = f"{self.model_name}_metrics_{timestamp}.json"
        _, _, self.test_loader = get_seg_data_loaders()
        self.load_model(model_path=model_path)

        cudnn.benchmark = True

    def load_model(self, model_path):
        log.info(f"Loading model weights from: {model_path}")
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device))

    def calculate_metrics(self, metrics=[]):
        results = {}
        results["metrics"] = {}

        for metric in metrics:
            if metric == "metrics_iou":
                value = self.get_average_iou()
            elif metric == "metrics_dice":
                value = self.get_average_dice()
            elif metric == "metrics_pixel-accuracy":
                value = self.get_average_p_acc()
            results["metrics"][metric.split("_")[0]] = value

        self.update_results_json(results)

    def get_average_iou(self):
        """
        Computes the average Intersection over Union (IoU) for the test dataset.

        Returns:
            float: The average IoU score.
        """
        total_iou = 0.0
        num_samples = 0
        self.model.eval()
        with torch.no_grad():
            for images, masks in self.test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                preds = self.model(images)
                preds = torch.argmax(preds, dim=1)  # Convert to class indices

                total_iou += iou(preds, masks)  # Compute IoU
                num_samples += 1

        return total_iou / num_samples if num_samples > 0 else 0.0

    def get_average_dice(self):
        """
        Computes the average Dice coefficient for the test dataset.

        Returns:
            float: The average Dice score.
        """
        total_dice = 0.0
        num_samples = 0
        self.model.eval()
        with torch.no_grad():
            for images, masks in self.test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                preds = self.model(images)
                preds = torch.argmax(preds, dim=1)  # Convert to class indices

                total_dice += dice(preds, masks)  # Compute Dice coefficient
                num_samples += 1

        return total_dice / num_samples if num_samples > 0 else 0.0

    def get_average_p_acc(self):
        """
        Computes the average Pixel accuracy for the test dataset.

        Returns:
            float: The average pixel accuracy.
        """
        total_dice = 0.0
        num_samples = 0
        self.model.eval()
        with torch.no_grad():
            for images, masks in self.test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                preds = self.model(images)
                preds = torch.argmax(preds, dim=1)  # Convert to class indices

                # Compute Pixel accuracy
                total_pixel_accuracy += pixel_accuracy(preds, masks)
                num_samples += 1

        return total_pixel_accuracy / num_samples if num_samples > 0 else 0.0

    def calculate_metrics_on_methods(self, methods=[]):
        """
        Iterates through the list of methods of perturbations to
        apply and calculates performance of the model on iou and
        dice metrics.
        """
        results = {}
        results["methods"] = {}
        log.info("Evaluating performance on perturbations.")

        for method in methods:
            results[method] = {}
            is_occlusion = False
            if method == "gaussian_noise":
                items = CONSTANTS["GAUSSIAN_NOISE_RANGE"]
                tfrm = gaussian_noise_transform
            elif method == "gaussian_blur":
                items = CONSTANTS["GAUSSIAN_BLUR_RANGE"]
                tfrm = gaussian_blur_transform
            elif method == "contrast_inc":
                items = CONSTANTS["CONTRAST_INC_RANGE"]
                tfrm = contrast_modify_transform
            elif method == "contrast_dec":
                items = CONSTANTS["CONTRAST_DEC_RANGE"]
                tfrm = contrast_modify_transform
            elif method == "brightness_inc":
                items = CONSTANTS["BRIGHTNESS_INC_RANGE"]
                tfrm = brightness_adjust_transform
            elif method == "brightness_dec":
                items = CONSTANTS["BRIGHTNESS_DEC_RANGE"]
                tfrm = brightness_adjust_transform
            elif method == "occlusion":
                items = CONSTANTS["OCCLUSION_SIZE_RANGE"]
                is_occlusion = True
            elif method == "s&p":
                items = CONSTANTS["S&P_NOISE_RANGE"]
                tfrm = s_and_p_noise_transform

            log.info(f"Applying {method} for ranges: {items}")

            for item in items:
                log.info(f"Current strength: {item}")
                results[method][item] = {}
                self.test_perturbation(
                    method, item, results, tfrm, is_occlusion)

            log.info(f"{method} at {item} over, updating JSON.")
            self.update_results_json(results)

    def test_perturbation(self, method, strn, results, transforms, is_occlusion=False):
        total_iou = 0.0
        total_dice = 0.0
        total_acc = 0.0
        num_samples = 0
        self.model.eval()

        with torch.no_grad():
            for images, masks in self.test_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                # If occlusion apply the method directly
                if is_occlusion:
                    images, masks = apply_occlusion(images, masks)
                else:
                    images = transforms(images)

                preds = self.model(images)
                preds = torch.argmax(preds, dim=1)  # Convert to class indices

                total_iou += iou(preds, masks)
                total_dice += dice(preds, masks)
                total_acc += pixel_accuracy(preds, masks)
                num_samples += 1

        m_iou = total_iou / num_samples
        m_dice = total_dice / num_samples
        m_p_acc = total_acc / num_samples

        results[method][strn]["iou"] = m_iou
        results[method][strn]["dice"] = m_dice
        results[method][strn]["pixel-accuracy"] = m_p_acc

    def update_results_json(self, current_dict):
        """
        Updates the results JSON file with new key-value pairs.

        Args:
            current_dict (dict): Dictionary containing new key-value pairs to be added.
        """
        # Check if file exists and load previous data
        if os.path.exists(self.results_file):
            with open(self.results_file, "r") as f:
                try:
                    results_data = json.load(f)
                except json.JSONDecodeError:
                    results_data = {}  # If file is empty or corrupted, reset
        else:
            results_data = {}

        # Update with new key-value pairs
        results_data.update(current_dict)

        # Write back to the file
        with open(self.results_file, "w") as f:
            json.dump(results_data, f, indent=4)

        log.info(f"Results JSON updated: {self.results_file}")


if __name__ == "__main__":
    allowed_models = ["unet",
                      "autoencoder_segmentation", "clip"]
    allowed_evals = ["metrics_iou", "metrics_dice", "metrics_pixel-accuracy", "contrast_inc", "contrast_dec", "occlusion",
                     "gaussian_noise", "s&p", "gaussian_blur", "brightness_inc", "brightness_dec"]
    parser = argparse.ArgumentParser(description="Evaluate model robustness")

    parser.add_argument("model_name", type=str,
                        help=f"Name of the model. Allowed options: {', '.join(allowed_models)}",
                        default="unet", choices=allowed_models)
    parser.add_argument("eval_methods", nargs="+",
                        help=f"List of eval methods. Allowed options: {', '.join(allowed_evals)}",
                        required=True)

    args = parser.parse_args()

    model_name = args.model_name
    eval_methods = args.eval_methods

    log.info(f"Model selected for evaluation: {model_name}")
    log.info(
        f"List of evaluations configured for running: {eval_methods}")

    model_path = load_selected_model(sub_dir=model_name)

    if model_name == "unet":
        model = UNet()
    elif model_name == "autoencoder_segmentation":
        model = AutoEncoderSegmentation()

    runner = EvaluationRunner(model_name=model_name,
                              model=model,
                              model_path=model_path)

    metrics_list = [i for i in eval_methods if "metrics" in i]
    methods_list = [i for i in eval_methods if "metrics" not in i]

    runner.calculate_metrics(metrics=metrics_list)
    runner.calculate_metrics_on_methods(methods=methods_list)
