import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage.util import random_noise


def _add_gaussian_pixel_noise(image, std_dev=0, noise_type='gaussian', noise_ratio=0.00):
    """
    Adds Gaussian or Salt & Pepper noise to a PyTorch image tensor.

    Args:
        image (torch.Tensor): Image tensor (C, H, W) with values in [0,1] or [0,255].
        std_dev (float): Standard deviation of Gaussian noise (only used if noise_type='gaussian').
        noise_type (str): Type of noise to apply ('gaussian' or 's&p').
        noise_ratio (float): Ratio of salt vs. pepper noise (only used if noise_type='s&p').

    Returns:
        torch.Tensor: Noisy image tensor in the same format.
    """
    image_np = (image.permute(1, 2, 0).cpu().numpy()) * 255.0

    if noise_type == 's&p':
        noisy_image = random_noise(
            image_np, mode="s&p", salt_vs_pepper=noise_ratio)
    else:  # Default to Gaussian noise
        noisy_image = random_noise(
            image_np, mode="gaussian", mean=0, var=(std_dev**2))

    processed_tensor = torch.tensor(
        noisy_image / 255.0).permute(2, 0, 1).to(image.dtype).to(image.device)

    return processed_tensor


def gaussian_noise_transform(std_dev=0):
    """
    Returns a torchvision transform applying Gaussian noise.

    Args:
        std_dev (float): Standard deviation of Gaussian noise.
        noise_type (str): Type of noise to apply ('gaussian' or 's&p').
        noise_ratio (float): Ratio of salt vs. pepper noise (used only if noise_type='s&p').

    Returns:
        torchvision.transforms.Lambda
    """
    return transforms.Lambda(lambda img: _add_gaussian_pixel_noise(
        image=img, std_dev=std_dev, noise_type='gaussian'))


def s_and_p_noise_transform(noise_ratio=0.00):
    """
    Returns a torchvision transform applying Salt & Pepper noise.

    Args:
        std_dev (float): Standard deviation of Gaussian noise.
        noise_type (str): Type of noise to apply ('gaussian' or 's&p').
        noise_ratio (float): Ratio of salt vs. pepper noise (used only if noise_type='s&p').

    Returns:
        torchvision.transforms.Lambda
    """
    return transforms.Lambda(lambda img: _add_gaussian_pixel_noise(
        image=img, noise_type='s&p', noise_ratio=noise_ratio))


def _apply_gaussian_blur(image, num_iterations=1):
    """
    Applies Gaussian blur using a 3x3 kernel multiple times to a PyTorch tensor.

    Args:
        image (torch.Tensor): Image tensor (C, H, W) in [0,1] or [0,255].
        num_iterations (int): Number of times to apply Gaussian blur.

    Returns:
        torch.Tensor: Blurred image tensor in the same format.
    """
    device = image.device  # Preserve original device

    # Convert to NumPy (H, W, C) and scale to [0, 255]
    image_np = (image.permute(1, 2, 0).cpu().numpy()) * 255.0

    # Convert to uint8 for OpenCV
    image_np = image_np.astype(np.uint8)

    # Apply Gaussian blur multiple times
    for _ in range(num_iterations):
        # 3x3 kernel, std dev auto
        image_np = cv2.GaussianBlur(image_np, (3, 3), 0)

    # Convert back to PyTorch tensor in [0,1] range
    blurred_tensor = torch.tensor(
        image_np / 255.0).permute(2, 0, 1).to(image.dtype).to(device)

    return blurred_tensor


def gaussian_blur_transform(num_iterations=1):
    """
    Returns a torchvision transform applying Gaussian blur.

    Args:
        num_iterations (int): Number of times to apply Gaussian blur.

    Returns:
        torchvision.transforms.Lambda
    """
    return transforms.Lambda(lambda img: _apply_gaussian_blur(img, num_iterations))


def _modify_contrast(image, contrast_factor=1.0):
    """
    Modifies image contrast by multiplying each pixel value by contrast_factor.

    Args:
        image (torch.Tensor): Image tensor (C, H, W) in [0,1] or [0,255].
        contrast_factor (float): Factor to multiply pixel values by.

    Returns:
        torch.Tensor: Contrast-adjusted image tensor in the same format.
    """
    device = image.device  # Preserve original device

    # Convert image to [0, 255] range
    image_scaled = (image * 255.0).clamp(0, 255)

    # Apply contrast adjustment
    # Cap values at 255
    image_contrasted = (image_scaled * contrast_factor).clamp(0, 255)

    # Convert back to [0,1] range and retain device & dtype
    contrast_tensor = (image_contrasted / 255.0).to(image.dtype).to(device)

    return contrast_tensor


def contrast_modify_transform(contrast_factor=1.0):
    """
    Returns a torchvision transform applying contrast increase.

    Args:
        contrast_factor (float): Factor to multiply pixel values by.

    Returns:
        torchvision.transforms.Lambda
    """
    return transforms.Lambda(lambda img: _modify_contrast(img, contrast_factor))


def _modify_brightness(image, brightness_offset=0):
    """
    Modifies image brightness by adding a fixed value to each pixel.

    Args:
        image (torch.Tensor): Image tensor (C, H, W) in [0,1] or [0,255].
        brightness_offset (float): Value to add to each pixel.

    Returns:
        torch.Tensor: Brightness-adjusted image tensor in the same format.
    """
    device = image.device  # Preserve original device

    # Convert image to [0, 255] range
    image_scaled = (image * 255.0).clamp(0, 255)

    # Apply brightness adjustment
    # Cap values at 255
    image_brightened = (image_scaled + brightness_offset).clamp(0, 255)

    # Convert back to [0,1] range and retain device & dtype
    brightened_tensor = (image_brightened / 255.0).to(image.dtype).to(device)

    return brightened_tensor


def brightness_adjust_transform(brightness_offset=0):
    """
    Returns a torchvision transform applying brightness modification.

    Args:
        brightness_offset (float): Value to add to each pixel.

    Returns:
        torchvision.transforms.Lambda
    """
    return transforms.Lambda(lambda img: _modify_brightness(img, brightness_offset))


def apply_occlusion(image_batch, mask_batch, occlusion_size=0):
    """
    Applies a random square occlusion to both image and mask batches.
    Works for batch processing.

    Args:
        image_batch (torch.Tensor): Image tensor (N, C, H, W) in [0,1] range.
        mask_batch (torch.Tensor): Mask tensor (N, H, W) in [0,1] or class indices.
        occlusion_size (int): Edge length of the occlusion square.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Occluded images and masks.
    """
    if occlusion_size == 0:
        return image_batch, mask_batch  # No occlusion applied

    n, c, h, w = image_batch.shape  # Get batch size and dimensions

    # Ensure occlusion size is within valid bounds
    occlusion_size = min(occlusion_size, h, w)

    # Clone inputs to avoid modifying original tensors
    image_batch = image_batch.clone()
    mask_batch = mask_batch.clone()

    for i in range(n):
        # Select a random top-left coordinate for occlusion (per image)
        x_start = np.random.randint(0, w - occlusion_size)
        y_start = np.random.randint(0, h - occlusion_size)

        # Apply occlusion (set pixels to 0)
        image_batch[i, :, y_start:y_start+occlusion_size,
                    x_start:x_start+occlusion_size] = 0
        mask_batch[i, y_start:y_start+occlusion_size,
                   x_start:x_start+occlusion_size] = 0

    return image_batch, mask_batch
