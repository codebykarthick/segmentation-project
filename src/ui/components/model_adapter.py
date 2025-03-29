from models.unet import UNet
from models.autoencoder_segmentation import AutoEncoderSegmentation
from models.clip_segmentation import ClipSegmentation
from PIL import Image
import torch
import torchvision.transforms as transforms
from util.constants import CONSTANTS

imageToTensor = transforms.ToTensor()


def create_model_and_segment(img_path: str, model_type: str, model_weight_path: str):
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
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    final_mask_path = None
    model = None
    tmp_dir = CONSTANTS["TMP_PATH"]

    # Create the model
    if model_type == "unet":
        model = UNet()
    elif model_type == "autoencoder_segmentation":
        model = AutoEncoderSegmentation()
    elif model_type == "clip_segmentation":
        model = ClipSegmentation()

    # Load the weights from the path provided
    model.load_state_dict(torch.load(
        model_weight_path, weights_only=True, map_location=device))

    # Run inference
    img = Image.open(img_path).convert("RGB")
    img_tensor = imageToTensor(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)

    mask_tensor = output.squeeze(0).cpu()

    return final_mask_path
