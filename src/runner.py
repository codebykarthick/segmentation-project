from datetime import datetime
import os
from models.autoencoder import Autoencoder
from models.autoencoder_segmentation import AutoEncoderSegmentation
from models.clip_segmentation import ClipSegmentation
from models.unet import UNet
import sys
import argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
from time import time
from util.constants import CONSTANTS
from util.cloud_tools import auto_shutdown
from util.data_loader import get_seg_data_loaders, get_data_loaders
from util.model_handler import load_selected_model
from util import logger

log = logger.setup_logger()


class Runner:
    """ Runner class for training and testing UNet and other models. """

    def __init__(self, model_name: str, model: torch.nn.Module, model_type: str = "seg") -> None:
        """ Initialize the Runner class with GPU support if available.

        Parameters:
            model_name (str): The name of the model.
            model (torch.nn.Module): The model instance to be trained or tested.
            model_type (str): The type of model ('seg' for segmentation or other for autoencoder). Default is 'seg'.

        Returns:
            None
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=CONSTANTS["LEARNING_RATE"])
        self.patience = 3
        self.counter = 0
        cudnn.benchmark = True

        self.type = model_type
        if self.type == "seg":
            log.info("Running a segmentation training.")
            self.train_loader, self.val_loader, self.test_loader = get_seg_data_loaders()
            self.criterion = nn.CrossEntropyLoss()
        else:
            log.info("Running an autoencoder training.")
            self.train_loader, self.val_loader, self.test_loader = get_data_loaders()
            self.criterion = nn.MSELoss()

    def train(self, epochs: int = 10) -> None:
        """ Train the model segmentation or autoencoder.

        Parameters:
            epochs (int): The number of epochs to train the model. Default is 10.

        Returns:
            None
        """
        if self.type == "seg":
            self.train_seg(epochs)
        else:
            self.train_autoencoder(epochs)

    def train_autoencoder(self, num_epochs: int = 10) -> None:
        """ Train the model and save the encoder and decoder weights for best epoch loops.

        Parameters:
            num_epochs (int): The number of epochs to train the autoencoder. Default is 10.

        Returns:
            None
        """
        self.model.train()
        val_loss = None
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            epoch_loss = 0
            scaler = torch.amp.GradScaler()
            toc = time()

            for batch_idx, images in enumerate(self.train_loader):

                images = images.to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast(self.device, dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, images)

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                scaler.step(self.optimizer)
                scaler.update()

                epoch_loss += loss.item()

                if (batch_idx + 1) % CONSTANTS["BATCH_LOG_FREQ"] == 0:
                    tic = time()
                    log.info(
                        f"Epoch: [{epoch+1}/{num_epochs}], Batch: [{batch_idx+1}/{len(self.train_loader)}], Loss: {(epoch_loss/(batch_idx + 1)):.4f}, Time: {(tic - toc):.2f}s")
                    toc = time()

            # Compute validation loss
            val_loss = self.validate()
            val_loss = val_loss if val_loss is not None else 0.0
            epoch_loss /= len(self.train_loader)

            log.info(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                self.counter = 0
                best_val_loss = val_loss
                # Save model whenever it is better than our current best
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.save_model(
                    f"{self.model_name}_{timestamp}_val_{val_loss:.4f}.pth")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    log.info(
                        f"Early stopping. Best val loss: {best_val_loss:.4f}.")
                    break

    def train_seg(self, num_epochs: int = 10) -> None:
        """ Train the model and save the weights for best epochs.

        Parameters:
            num_epochs (int): The number of epochs to train the segmentation model. Default is 10.

        Returns:
            None
        """
        self.model.train()
        val_loss = None
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            epoch_loss = 0
            scaler = torch.amp.GradScaler()

            toc = time()
            for batch_idx, (images, masks) in enumerate(self.train_loader):

                images, masks = images.to(self.device), masks.to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast(self.device, dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                scaler.step(self.optimizer)
                scaler.update()

                epoch_loss += loss.item()

                if (batch_idx + 1) % CONSTANTS["BATCH_LOG_FREQ"] == 0:
                    tic = time()
                    log.info(
                        f"Epoch: [{epoch+1}/{num_epochs}], Batch: [{batch_idx+1}/{len(self.train_loader)}], Loss: {(epoch_loss / (batch_idx + 1)):.4f}, Time: {(tic - toc):.2f}s")
                    toc = time()

            # Compute validation loss
            val_loss = self.validate()
            val_loss = val_loss if val_loss is not None else 0.0
            epoch_loss /= len(self.train_loader)

            log.info(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                self.counter = 0
                best_val_loss = val_loss
                # Save model whenever it is better than our current best
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.save_model(
                    f"{self.model_name}_{timestamp}_val_{val_loss:.4f}.pth")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    log.info(
                        f"Early stopping. Best val loss: {best_val_loss:.4f}.")
                    break

    def validate(self) -> float:
        """ Validate the model against the validation set.

        Returns:
            float: The average validation loss.
        """
        if self.type == "seg":
            return self.validate_segmentation()
        else:
            return self.validate_autoencoder()

    def validate_segmentation(self) -> float:
        """ Compute validation loss for segmentation.

        Returns:
            float: The average segmentation loss.
        """
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        num_batches = len(self.val_loader)

        if num_batches == 0:
            log.warning("Validation loader is empty. Skipping validation.")
            return 0

        with torch.no_grad():  # Disable gradient calculation
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

        return total_loss / num_batches

    def validate_autoencoder(self) -> float:
        """ Compute validation loss for autoencoder.

        Returns:
            float: The average autoencoder loss.
        """
        self.model.eval()
        total_loss = 0

        num_batches = len(self.val_loader)

        if num_batches == 0:
            log.warning("Validation loader is empty. Skipping validation.")
            return 0

        with torch.no_grad():  # Disable gradient calculation
            for images in self.val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, images)
                total_loss += loss.item()

        return total_loss / num_batches

    def test(self, model_path: str) -> None:
        """ Test the model.

        Parameters:
            model_path (str): The path to the model weights to be tested.

        Returns:
            None
        """
        if self.type == 'seg':
            self.test_segmentation(model_path)
        else:
            self.test_autoencoder(model_path)

    def test_autoencoder(self, model_path: str) -> None:
        """ Test the autoencoder model.

        Parameters:
            model_path (str): The path to the model weights to be tested.

        Returns:
            None
        """
        if model_path is None:
            log.error("Model path not provided.")
            return

        self.load_model(model_path)
        log.info("Loaded model for test dataset.")
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_id, (images) in enumerate(self.test_loader):
                images = images.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, images)
                total_loss += loss.item()

                if (batch_id + 1) % CONSTANTS["BATCH_LOG_FREQ"] == 0:
                    log.info(
                        f"Batches evaluated: [{batch_id + 1}/{len(self.test_loader)}], current test loss: {(total_loss/(batch_id + 1)):.4f}")

            total_loss /= len(self.test_loader)
        log.info(f"Avg Test Loss: {total_loss:.4f}")

    def test_segmentation(self, model_path: str) -> None:
        """ Test the segmentation model.

        Parameters:
            model_path (str): The path to the model weights to be tested.

        Returns:
            None
        """
        if model_path is None:
            log.error("Model path not provided.")
            return

        self.load_model(model_path)
        log.info("Loaded model for test dataset.")
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_id, (images, masks) in enumerate(self.test_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

                if (batch_id + 1) % CONSTANTS["BATCH_LOG_FREQ"] == 0:
                    log.info(
                        f"Batches evaluated: [{batch_id + 1}/{len(self.test_loader)}], current test loss: {total_loss/(batch_id + 1)}")
            total_loss /= len(self.test_loader)
        log.info(f"Avg Test Loss: {total_loss:.4f}")

    def save_model(self, file_name: str = "unet_checkpoint.pth") -> None:
        """ Save model weights.

        Parameters:
            file_name (str): The name of the file to save the model weights. Default is "unet_checkpoint.pth".

        Returns:
            None
        """
        if not os.path.exists(CONSTANTS["WEIGHTS_PATH"]):
            os.makedirs(CONSTANTS["WEIGHTS_PATH"])

        actual_path = os.path.join(CONSTANTS["WEIGHTS_PATH"], self.model_name)
        if not os.path.exists(actual_path):
            os.makedirs(actual_path)

        if self.model_name == "autoencoder_segmentation":
            # Save encoder weights separately
            torch.save(self.model.encoder.state_dict(),
                       os.path.join(actual_path, "encoder_" + file_name))
            torch.save(self.model.decoder.state_dict(),
                       os.path.join(actual_path, "decoder_" + file_name))
        else:
            torch.save(self.model.state_dict(),
                       os.path.join(actual_path, file_name))

    def load_model(self, file_name: str) -> None:
        """ Load model weights.

        Parameters:
            file_name (str): The name of the file containing the model weights.

        Returns:
            None
        """

        if self.model_name == "autoencoder_segmentation":
            # Load encoder weights separately
            encoder_file = os.path.join(
                CONSTANTS["WEIGHTS_PATH"], self.model_name, "encoder_" + file_name)
            decoder_file = os.path.join(
                CONSTANTS["WEIGHTS_PATH"], self.model_name, "decoder_" + file_name)
            if not os.path.exists(encoder_file) or not os.path.exists(decoder_file):
                log.error(
                    "Model weights not found: {encoder_file}, {decoder_file}")
                return

            self.model.encoder.load_state_dict(
                torch.load(encoder_file, weights_only=True, map_location=self.device))
            self.model.decoder.load_state_dict(
                torch.load(decoder_file, weights_only=True, map_location=self.device))
            self.model.eval()
            return

        file_path = os.path.join(CONSTANTS["WEIGHTS_PATH"], file_name)
        if not os.path.exists(file_path):
            log.error(f"Model weights not found: {file_path}")
            return

        self.model.load_state_dict(torch.load(
            os.path.join(CONSTANTS["WEIGHTS_PATH"], file_name), weights_only=True, map_location=self.device))
        self.model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model training or testing.")
    parser.add_argument("--model_name", type=str,
                        help="Name of the model to use.")
    parser.add_argument("--mode", type=str, choices=[
                        "train", "test"], help="Operation mode: train or test.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for training (only used in train mode).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Size of batch to be used for training, validation and testing")
    parser.add_argument("--env", type=str, choices=["local", "cloud"], default="local",
                        help="Cloud mode has a special shutdown sequence to save resources.")
    parser.add_argument("--copy_dir", type=str,
                        help="Directory where logs and weights folder will be copied (required if env is cloud)")
    args = parser.parse_args()
    if args.env == "cloud":
        if not args.copy_dir:
            parser.error("--copy_dir is required when env is cloud")
        elif not os.path.exists(args.copy_dir):
            log.error("The copy directory does not exist, halting execution.")
            sys.exit(1)

    model_name = args.model_name.lower()
    mode = args.mode.lower()
    model_type = "seg"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "unet":
        model = UNet(in_channels=3, out_channels=3)
    elif model_name == "autoencoder":
        model = Autoencoder()
        model_type = "auto"
    elif model_name == "autoencoder_segmentation":
        selected_encoder = load_selected_model(sub_dir="autoencoder")
        if selected_encoder:
            encoder = Autoencoder()
            encoder.load_state_dict(torch.load(
                os.path.join(CONSTANTS["WEIGHTS_PATH"],
                             "autoencoder", selected_encoder),
                weights_only=True, map_location=device
            ))
            model = AutoEncoderSegmentation(pretrained_encoder=encoder)
    elif model_name == "clip_segmentation":
        model = ClipSegmentation(device=device)
    else:
        log.error(
            "Invalid model name. Supported: unet, autoencoder, autoencoder_segmentation")
        sys.exit(1)

    # Initialize Runner with the model chosen
    runner = Runner(model_name=model_name, model=model, model_type=model_type)

    if mode == "train":
        log.info(f"Training and validating {model_name} model")
        runner.train(epochs=args.epochs)
    elif mode == "test":
        log.info(f"Evaluating trained {model_name} model on test set")
        selected_model = load_selected_model(sub_dir=model_name)
        if selected_model:
            runner.test(selected_model)
    else:
        log.error("Invalid mode provided. Use 'train' or 'test'.")
        sys.exit(1)

    if args.env == "cloud":
        auto_shutdown(args.copy_dir)
