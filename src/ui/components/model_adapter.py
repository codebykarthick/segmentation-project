from datetime import datetime
from models.unet import UNet
from models.autoencoder_segmentation import AutoEncoderSegmentation
from models.clip_segmentation import ClipSegmentation
from models.prompt_segmentation import PromptSegmentation
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from util.constants import CONSTANTS

imageToTensor = transforms.ToTensor()


def create_model_and_segment(img_path: str, model_type: str, model_weight_path: str, prompt_mode: bool, prompt):
    """
    Get the input image and run inference, generate mask. Save said mask as a temporary
    file.

    Args:
        img_path: Path to the image to infer
        model_type: Type of the model instance to laod
        model_weight_path: Path of the model weights to load into the model instance

    Returns:
        Returns the path of 

    Return path of the temporary file saved to be loaded for final preview.
    """
    # Get the supported device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    tmp_dir = CONSTANTS["TMP_PATH"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_mask_file_name = f"mask_{timestamp}.png"
    tmp_path = os.path.join(os.getcwd(), tmp_dir)
    final_mask_path = os.path.join(tmp_path, final_mask_file_name)

    # Create the model
    if model_type == "unet":
        model = UNet()
    elif model_type == "autoencoder_segmentation":
        model = AutoEncoderSegmentation()
    elif model_type == "clip_segmentation":
        model = ClipSegmentation()
    elif model_type == "prompt_segmentation":
        model = PromptSegmentation()

    # Load the weights from the path provided
    model.load_state_dict(torch.load(
        model_weight_path, weights_only=True, map_location=device))
    model = model.to(device)

    # Run inference
    img = Image.open(img_path).convert("RGB")
    if prompt_mode:
        assert prompt is not None, "Prompt must be provided in prompt mode"
        prompt_tensor = torch.tensor(prompt, dtype=torch.float32)
        if prompt_tensor.ndim == 2:
            prompt_tensor = prompt_tensor.unsqueeze(0)
        img_tensor = imageToTensor(img)
        img_tensor = torch.cat((img_tensor, prompt_tensor),
                               dim=0).unsqueeze(0).to(device)
    else:
        img_tensor = imageToTensor(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)

    mask_tensor = output.squeeze(0).cpu()
    # Convert logits to class predictions
    softmax = torch.nn.Softmax(dim=0)
    probs = softmax(mask_tensor)
    predicted_classes = probs.argmax(0).cpu().numpy()

    # Create a color map: background=black, cat=red, dog=green
    color_map = {
        0: (0, 0, 0),       # Background - Black
        1: (255, 0, 0),     # Cat - Red
        2: (0, 255, 0),     # Dog - Green
    }

    # Create RGB image
    height, width = predicted_classes.shape
    rgb_image = Image.new("RGB", (width, height))
    pixels = rgb_image.load()

    for y in range(height):
        for x in range(width):
            pixels[x, y] = color_map[predicted_classes[y, x]]

    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    rgb_image.save(final_mask_path)

    return final_mask_path
