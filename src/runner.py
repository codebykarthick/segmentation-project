import curses
from datetime import datetime
import os
from models.unet import UNet
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from util.data_loader import get_data_loaders
from util import logger

WEIGHTS_PATH = "weights"
log = logger.setup_logger()


class Runner:
    """ Runner class for training and testing UNet. """

    def __init__(self, model_name, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        """ Initialize the Runner class, with GPU support if available. """
        self.model_name = model_name
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders()
        self.WEIGHTS_PATH = WEIGHTS_PATH

    def train(self, num_epochs=10):
        """ Train the model. """
        self.model.train()
        val_loss = None
        for epoch in range(num_epochs):
            epoch_loss = 0
            for images, masks in self.train_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Compute validation loss
            val_loss = self.validate()
            val_loss = val_loss if val_loss is not None else 0.0

            log.info(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
        # Save model with model name and timestamp at the end of training
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_model(
            f"{self.model_name}_{timestamp}_val_{val_loss:.4f}.pth")

    def validate(self):
        """ Compute validation loss. """
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

    def test(self, model_path):
        """ Test the model. """
        if model_path is None:
            log.error("Model path not provided.")
            return

        self.load_model(model_path)
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, masks in self.test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
        log.info(f"Test Loss: {total_loss:.4f}")

    def save_model(self, file_name="unet_checkpoint.pth"):
        """ Save model weights. """
        if not os.path.exists(self.WEIGHTS_PATH):
            os.makedirs(self.WEIGHTS_PATH)

        torch.save(self.model.state_dict(), os.path.join(
            self.WEIGHTS_PATH, file_name))

    def load_model(self, file_name):
        """ Load model weights. """
        if not os.path.exists(os.path.join(self.WEIGHTS_PATH, file_name)):
            log.error("Model weights not found.")
            return

        self.model.load_state_dict(torch.load(
            os.path.join(self.WEIGHTS_PATH, file_name)))
        self.model.eval()


def list_model_weights(extension=".pth"):
    """ List all available model weights in the given directory. """
    if not os.path.exists(WEIGHTS_PATH):
        log.error("No weights directory found.")
        return []
    return [f for f in sorted(os.listdir(WEIGHTS_PATH)) if f.endswith(extension)]


def select_model_weight(stdscr, weights):
    """ Interactive model selection using arrow keys. """
    curses.curs_set(0)  # Hide cursor
    idx = 0  # Start selection at first item

    while True:
        stdscr.clear()
        stdscr.addstr("Select a model weight file:\n", curses.A_BOLD)

        # Display weight files
        for i, weight in enumerate(weights):
            if i == idx:
                # Highlight selected
                stdscr.addstr(f"> {weight}\n", curses.A_REVERSE)
            else:
                stdscr.addstr(f"  {weight}\n")

        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_UP and idx > 0:
            idx -= 1
        elif key == curses.KEY_DOWN and idx < len(weights) - 1:
            idx += 1
        elif key in [curses.KEY_ENTER, 10, 13]:  # Enter key pressed
            return weights[idx]


def load_selected_model():
    """ Display model weights and let user select one. """
    weights = list_model_weights()

    if not weights:
        log.error("No model weights found!")
        return None

    # Launch interactive selection
    selected_weight = curses.wrapper(select_model_weight, weights)
    log.info(f"Selected model: {selected_weight}")

    return selected_weight


if __name__ == "__main__":
    if len(sys.argv) < 3:
        log.error("Usage: python runner.py <model_name> <mode>")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    mode = sys.argv[2].lower()
    epochs = int(sys.argv[3])

    if model_name == "unet":
        model = UNet(in_channels=3, out_channels=3)
        runner = Runner(model_name=model_name, model=model)
    else:
        log.error("Invalid model name. Supported: unet")
        sys.exit(1)

    if mode == "train":
        log.info("Training and validating model")
        runner.train(epochs=epochs)
    elif mode == "test":
        log.info("Evaluating trained model on test set")
        selected_model = load_selected_model()
        if selected_model:
            runner.test(os.path.join(WEIGHTS_PATH, selected_model))
    else:
        log.error("Invalid mode provided. Use 'train' or 'test'.")
        sys.exit(1)
