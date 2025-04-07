import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage.util import random_noise
from typing import Tuple


def _add_gaussian_pixel_noise(image: torch.Tensor, std_dev: float = 0.0, noise_type: str = 'gaussian', noise_ratio: float = 0.0, amount: float = 0.05) -> torch.Tensor:
    """
    Adds noise to a PyTorch image tensor.

    Args:
        image (torch.Tensor): Image tensor with shape (C, H, W) and pixel values in [0, 1].
        std_dev (float, optional): Standard deviation for Gaussian noise in pixel scale (0–255). Defaults to 0.0.
        noise_type (str, optional): 'gaussian' or 's&p'. Defaults to 'gaussian'.
        noise_ratio (float, optional): Salt-vs-pepper ratio for S&P noise. Defaults to 0.0.
        amount (float, optional): Fraction of pixels to alter for S&P. Defaults to 0.05.

    Returns:
        torch.Tensor: Image tensor with noise added, in same shape and dtype, pixel values in [0, 1].
    """
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = image_np * 255.0  # Convert to pixel scale

    if noise_type == 's&p':
        print(
            f"Adding salt and pepper noise with ratio: {noise_ratio}, amount: {amount}")
        h, w, c = image_np.shape

        # Create grayscale S&P mask
        mask = random_noise(np.mean(image_np, axis=2) / 255.0,
                            mode='s&p', amount=amount, salt_vs_pepper=noise_ratio)
        mask = np.clip(mask * 255.0, 0, 255)

        # Broadcast to RGB
        noisy_image = np.stack([mask] * c, axis=2)
    else:
        print(f"Adding gaussian noise of std: {std_dev}")
        noise = np.random.normal(0, std_dev, size=image_np.shape)
        noisy_image = image_np + noise
        noisy_image = np.clip(noisy_image, 0, 255)

    # Convert back to tensor in [0, 1]
    noisy_image = torch.tensor(
        noisy_image / 255.0).permute(2, 0, 1).to(image.dtype).to(image.device)
    return noisy_image


def gaussian_noise_transform(std_dev: float = 0.0) -> transforms.Lambda:
    """
    Creates a transform that applies Gaussian noise to an image tensor.

    Args:
        std_dev (float, optional): Standard deviation for the Gaussian noise. Defaults to 0.0.

    Returns:
        torchvision.transforms.Lambda: A transform that applies Gaussian noise.
    """
    return transforms.Lambda(lambda img: _add_gaussian_pixel_noise(
        image=img, std_dev=std_dev, noise_type='gaussian'))


def s_and_p_noise_transform(noise_ratio: float = 0.00, amount: float = 0.05) -> transforms.Lambda:
    """
    Creates a transform that applies salt & pepper noise to an image tensor.

    Args:
        noise_ratio (float, optional): Ratio of salt vs. pepper noise. Defaults to 0.00.
        amount (float, optional): Amount of salt and pepper noise to add. Defaults to 0.05.

    Returns:
        torchvision.transforms.Lambda: A transform that applies salt & pepper noise.
    """
    return transforms.Lambda(lambda img: _add_gaussian_pixel_noise(
        image=img, noise_type='s&p', noise_ratio=noise_ratio, amount=amount))


def _apply_gaussian_blur(image: torch.Tensor, num_iterations: int = 1) -> torch.Tensor:
    """
    Applies Gaussian blur using a 3x3 kernel multiple times on an image tensor.

    Args:
        image (torch.Tensor): Image tensor with shape (C, H, W) with values in [0, 1] or [0, 255].
        num_iterations (int, optional): Number of iterations to apply the blur. Defaults to 1.

    Returns:
        torch.Tensor: The blurred image tensor in the same format as the input.
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


def gaussian_blur_transform(num_iterations: int = 1) -> transforms.Lambda:
    """
    Creates a transform that applies Gaussian blur to an image tensor.

    Args:
        num_iterations (int, optional): Number of times to apply the Gaussian blur. Defaults to 1.

    Returns:
        torchvision.transforms.Lambda: A transform that applies Gaussian blur.
    """
    return transforms.Lambda(lambda img: _apply_gaussian_blur(img, num_iterations))


def _modify_contrast(image: torch.Tensor, contrast_factor: float = 1.0) -> torch.Tensor:
    """
    Adjusts the contrast of an image tensor by scaling its pixel values.

    Args:
        image (torch.Tensor): Image tensor with shape (C, H, W) with values in [0, 1] or [0, 255].
        contrast_factor (float, optional): Factor by which to multiply the pixel values. Defaults to 1.0.

    Returns:
        torch.Tensor: The contrast-adjusted image tensor in the same format as the input.
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


def contrast_modify_transform(contrast_factor: float = 1.0) -> transforms.Lambda:
    """
    Creates a transform that adjusts the contrast of an image tensor.

    Args:
        contrast_factor (float, optional): Factor by which to adjust the contrast. Defaults to 1.0.

    Returns:
        torchvision.transforms.Lambda: A transform that applies contrast modification.
    """
    return transforms.Lambda(lambda img: _modify_contrast(img, contrast_factor))


def _modify_brightness(image: torch.Tensor, brightness_offset: float = 0) -> torch.Tensor:
    """
    Adjusts the brightness of an image tensor by adding a fixed offset to its pixel values.

    Args:
        image (torch.Tensor): Image tensor with shape (C, H, W) with values in [0, 1] or [0, 255].
        brightness_offset (float, optional): Value to add to each pixel. Defaults to 0.

    Returns:
        torch.Tensor: The brightness-adjusted image tensor in the same format as the input.
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


def brightness_adjust_transform(brightness_offset: float = 0) -> transforms.Lambda:
    """
    Creates a transform that adjusts the brightness of an image tensor.

    Args:
        brightness_offset (float, optional): The offset to add to each pixel. Defaults to 0.

    Returns:
        torchvision.transforms.Lambda: A transform that applies brightness adjustment.
    """
    return transforms.Lambda(lambda img: _modify_brightness(img, brightness_offset))


def apply_occlusion(image_batch: torch.Tensor, mask_batch: torch.Tensor, occlusion_size: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a random square occlusion to batches of images and masks.

    Args:
        image_batch (torch.Tensor): Batch of images with shape (N, C, H, W) in the [0, 1] range.
        mask_batch (torch.Tensor): Batch of masks with shape (N, H, W) containing either normalized values or class indices.
        occlusion_size (int, optional): Edge length of the square occlusion. Defaults to 0 (no occlusion).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The occluded image and mask batches.
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
        x_start = np.random.randint(0, max(1, w - occlusion_size + 1))
        y_start = np.random.randint(0, max(1, h - occlusion_size + 1))

        # Apply occlusion (set pixels to 0)
        image_batch[i, :, y_start:y_start+occlusion_size,
                    x_start:x_start+occlusion_size] = 0
        mask_batch[i, y_start:y_start+occlusion_size,
                   x_start:x_start+occlusion_size] = 0

    return image_batch, mask_batch
