import os
import argparse
from datetime import datetime
from evaluation.metrics import iou, dice, pixel_accuracy
from evaluation.robustness import gaussian_noise_transform, gaussian_blur_transform, contrast_modify_transform, brightness_adjust_transform, apply_occlusion, s_and_p_noise_transform
from models.unet import UNet
from models.autoencoder import Autoencoder
from models.autoencoder_segmentation import AutoEncoderSegmentation
from models.clip_segmentation import ClipSegmentation
from models.prompt_segmentation import PromptSegmentation
import torch
import torch.backends.cudnn as cudnn
from util import logger
from util.constants import CONSTANTS
from util.data_loader import get_seg_data_loaders
from util.model_handler import load_selected_model
import json
from tqdm import tqdm
from typing import Callable, Dict, Any, List, Union

log = logger.setup_logger()


class EvaluationRunner:
    """
    Evaluation Runner class for evaluating the model against
    several metrics and robustness measures.
    """

    def __init__(self, model_name: str, model: torch.nn.Module, model_path: str, batch_size: int = 8) -> None:
        """
        Initialize the EvaluationRunner with the specified model and load its weights.

        Parameters:
            model_name (str): Name of the model.
            model (torch.nn.Module): The model instance for evaluation.
            model_path (str): Path to the saved model weights.
            batch_size (int): Size of the batch for testing.

        Returns:
            None
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = model.to(self.device)
        self.load_model(model_path=model_path)

        if "prompt" in self.model_name:
            self.prompt = True
        else:
            self.prompt = False

        _, _, self.test_loader = get_seg_data_loaders(
            batch_size=batch_size, prompt_mode=self.prompt)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.results_file = f"results_{timestamp}.json"

        cudnn.benchmark = True

    def load_model(self, model_path: str) -> None:
        """
        Load model weights from the specified path.

        Parameters:
            model_path (str): Path to the model weights file.

        Returns:
            None
        """
        log.info(f"Loading model weights from: {model_path}")
        self.model.load_state_dict(torch.load(
            model_path, weights_only=True, map_location=self.device))

    def calculate_metrics(self, metrics: List[str] = []) -> None:
        """
        Calculate evaluation metrics on the test set without any perturbations.

        Parameters:
            metrics (List[str]): List of metric identifiers to calculate. E.g., 'metrics_iou', 'metrics_dice', 'metrics_pixel-accuracy'.

        Returns:
            None
        """
        results = {}
        results["metrics"] = {}

        for metric in metrics:
            if metric == "metrics_iou":
                log.info(
                    "Calculating the average IoU on test set without any perturbations.")
                values = self.get_average_metric(iou)
            elif metric == "metrics_dice":
                log.info(
                    "Calculating the average dice coefficient on test set without any perturbations.")
                values = self.get_average_metric(dice)
            elif metric == "metrics_pixel-accuracy":
                log.info(
                    "Calculating the average pixel accuracy on test set without any perturbations.")
                values = self.get_average_metric(pixel_accuracy)
            results["metrics"][metric.split("_")[1]] = {
                "background": values.get(0, 0),
                "cat": values.get(1, 0),
                "dog": values.get(2, 0),
                "mean": (sum(value for value in values.values() if value is not None) / len(values)) if values else 0
            }

        self.update_results_json(results)

    def get_average_metric(self, metric_fn: Callable, trs=None, is_occlusion: bool = False) -> Dict[Any, float]:
        """
        Compute the average value for a given metric function (IoU, Dice, or Pixel Accuracy) over the test set.

        Args:
            metric_fn (Callable): The metric function to apply.
            transforms: Any transforms to apply to the images.
            is_occlusion (bool): Whether to apply occlusion to the images.

        Returns:
            Dict[Any, float]: A dictionary containing the average metric score for each class.
        """
        total_metric = {}  # Dictionary to store metric for each class
        num_samples = {}  # Dictionary to track number of samples per class
        self.model.eval()
        with torch.no_grad():
            for images, masks in tqdm(self.test_loader, desc='Processing batches'):
                images, masks = images.to(self.device), masks.to(self.device)

                # Apply occlusion or transforms if specified
                if is_occlusion:
                    images, masks = apply_occlusion(images, masks)
                elif trs:
                    images = torch.stack([trs(img) for img in images])

                preds = self.model(images)
                preds = torch.softmax(preds, dim=1)
                preds = torch.argmax(preds, dim=1)

                for i in range(masks.shape[0]):
                    metric_scores = metric_fn(preds[i], masks[i])
                    for cls, score in metric_scores.items():
                        total_metric[cls] = total_metric.get(cls, 0) + score
                    for cls in metric_scores.keys():
                        num_samples[cls] = num_samples.get(cls, 0) + 1

        return {cls: total_metric[cls] / num_samples[cls] for cls in total_metric if cls in num_samples} if num_samples else {}

    def calculate_metrics_on_methods(self, methods: List[str] = []) -> None:
        """
        Evaluate the model performance on various perturbation methods by calculating metrics like IoU and Dice.

        Parameters:
            methods (List[str]): List of perturbation method identifiers (e.g., 'gaussian_noise', 'occlusion').

        Returns:
            None
        """
        if len(methods) == 0:
            return
        results = {}
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
            elif method == "sandp":
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

    def test_perturbation(self, method: str, strn: str, results: dict, trs=None, is_occlusion: bool = False) -> None:
        """
        Test a perturbation method by applying it to the test set and computing metrics.

        Parameters:
            method (str): The perturbation method identifier.
            strn (Union[str, int]): The current strength or level of perturbation.
            results (dict): Dictionary to store the resulting metrics.
            transforms (Optional[Callable]): Transform function to apply for the perturbation.
            is_occlusion (bool): Whether to apply occlusion.

        Returns:
            None
        """
        # m_iou = self.get_average_metric(iou, transforms, is_occlusion)
        m_dice = self.get_average_metric(dice, trs(strn), is_occlusion)
        # m_p_acc = self.get_average_metric(
        # pixel_accuracy, transforms, is_occlusion)

        # results[method][strn]["iou"] = m_iou
        results[method][strn]["dice"] = m_dice
        # results[method][strn]["pixel-accuracy"] = m_p_acc

    def update_results_json(self, current_dict: dict) -> None:
        """
        Update the results JSON file with new key-value pairs.

        Parameters:
            current_dict (dict): Dictionary containing new key-value pairs to be added.

        Returns:
            None
        """
        if not os.path.exists("results"):
            os.makedirs("results")
        results_path = os.path.join("results", self.results_file)
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                try:
                    results_data = json.load(f)
                except json.JSONDecodeError:
                    results_data = {}
        else:
            results_data = {}

        results_data.update(current_dict)

        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=4)

        log.info(f"Current results: {json.dumps(results_data, indent=4)}")
        log.info(f"Results saved at: {results_path}")


if __name__ == "__main__":
    allowed_models = ["unet", "autoencoder_segmentation",
                      "clip_segmentation", "prompt_segmentation"]
    allowed_evals = ["metrics_iou", "metrics_dice", "metrics_pixel-accuracy", "contrast_inc", "contrast_dec", "occlusion",
                     "gaussian_noise", "sandp", "gaussian_blur", "brightness_inc", "brightness_dec"]
    parser = argparse.ArgumentParser(description="Evaluate model robustness")

    parser.add_argument("--model_name", type=str,
                        help=f"Name of the model. Allowed options: {', '.join(allowed_models)}",
                        default="unet", choices=allowed_models)
    parser.add_argument("--batch_size", type=int,
                        help="Size of the batch to be used for the dataloader for metrics.")
    parser.add_argument("--eval_methods", nargs="+",
                        help=f"List of eval methods. Allowed options: {', '.join(allowed_evals)}")

    args = parser.parse_args()

    model_name = args.model_name
    eval_methods = args.eval_methods
    batch_size = args.batch_size

    log.info(f"Model selected for evaluation: {model_name}")
    log.info(
        f"List of evaluations configured for running: {eval_methods}")

    model_path = load_selected_model(sub_dir=model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "unet":
        model = UNet()
    elif model_name == "autoencoder_segmentation":
        selected_encoder = load_selected_model(
            sub_dir="autoencoder", filters=["encoder"])
        if selected_encoder:
            autoencoder = Autoencoder()
            # Get the encoder half alone for the segmentation task
            encoder = autoencoder.encoder
            encoder.load_state_dict(torch.load(
                os.path.join(CONSTANTS["WEIGHTS_PATH"],
                             "autoencoder", selected_encoder),
                weights_only=True, map_location=device
            ))
            model = AutoEncoderSegmentation(pretrained_encoder=encoder)
    elif model_name == "clip_segmentation":
        model = ClipSegmentation()
    elif model_name == "prompt_segmentation":
        model = PromptSegmentation()

    runner = EvaluationRunner(model_name=model_name,
                              model=model,
                              model_path=model_path, batch_size=batch_size)

    metrics_list = [i for i in eval_methods if "metrics" in i]
    methods_list = [i for i in eval_methods if "metrics" not in i]

    runner.calculate_metrics(metrics=metrics_list)
    runner.calculate_metrics_on_methods(methods=methods_list)
