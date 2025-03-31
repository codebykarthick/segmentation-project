from collections import defaultdict
import torch
from typing import Dict

# TODO: Review the metrics evaluation logic


def iou(pred_mask: torch.Tensor, actual_mask: torch.Tensor, num_classes: int = 3) -> Dict[int, float]:
    """
    Compute Intersection over Union (IoU) for each segmentation class.

    This function calculates the IoU for each segmentation class present in the predicted and ground truth masks.
    The IoU is computed as the ratio of the intersection area to the union area for each class.

    Args:
        pred_mask (torch.Tensor): The predicted segmentation mask with shape [H, W].
        actual_mask (torch.Tensor): The ground truth segmentation mask with shape [H, W].
        num_classes (int, optional): Total number of segmentation classes (including background). Defaults to 3.

    Returns:
        Dict[int, float]: A dictionary mapping each class index to its IoU score.
    """
    iou_scores = {}

    for class_idx in range(num_classes):
        pred_class = (pred_mask == class_idx)
        actual_class = (actual_mask == class_idx)

        intersection = torch.sum(pred_class & actual_class, dim=(-2, -1))
        union = torch.sum(pred_class | actual_class, dim=(-2, -1))

        # Skip classes that actual class does not even have label for
        non_empty_mask = (actual_class != 0)
        if non_empty_mask.any():
            # Compute IoU only for images (or pixels) where this class actually appears
            class_iou = intersection / (union + 1e-6)
            iou_scores[class_idx] = class_iou.item()

    return iou_scores


def dice(pred_mask: torch.Tensor, actual_mask: torch.Tensor, num_classes: int = 3) -> Dict[int, float]:
    """
    Compute the Dice coefficient for each segmentation class.

    This function computes the Dice coefficient, a measure of overlap between the predicted and ground truth masks,
    for each class (excluding background).

    Args:
        pred_mask (torch.Tensor): The predicted segmentation mask with shape [H, W] or [N, H, W].
        actual_mask (torch.Tensor): The ground truth segmentation mask with shape [H, W] or [N, H, W].
        num_classes (int, optional): Number of segmentation classes (excluding background). Defaults to 3.

    Returns:
        Dict[int, float]: A dictionary mapping each class index to its Dice coefficient.
    """
    dice_scores = {}

    for class_idx in range(num_classes):  # Skip background class (0)
        pred_class = (pred_mask == class_idx).to(torch.float32)
        actual_class = (actual_mask == class_idx).to(torch.float32)

        intersection = torch.sum(pred_class * actual_class, dim=(-2, -1))
        total_pixels = torch.sum(pred_class, dim=(-2, -1)) + \
            torch.sum(actual_class, dim=(-2, -1))

        if torch.sum(actual_class) > 0:
            class_dice = (2 * intersection) / (total_pixels + 1e-6)
            dice_scores[class_idx] = class_dice.item()

    return dice_scores


def pixel_accuracy(pred_mask: torch.Tensor, actual_mask: torch.Tensor, num_classes: int = 3) -> Dict[int, float]:
    """
    Compute the pixel-wise accuracy for each segmentation class.

    This function calculates the accuracy by comparing the predicted mask with the ground truth mask on a per-class basis.
    For each class, it computes the ratio of correctly predicted pixels to the total number of pixels for that class.

    Args:
        pred_mask (torch.Tensor): The predicted segmentation mask with shape [N, H, W].
        actual_mask (torch.Tensor): The ground truth segmentation mask with shape [N, H, W].
        num_classes (int, optional): Total number of segmentation classes. Defaults to 3.

    Returns:
        Dict[int, float]: A dictionary mapping each class index to its pixel-wise accuracy.
    """
    accuracy_scores = {}

    for class_idx in range(num_classes):
        pred_class = (pred_mask == class_idx)
        actual_class = (actual_mask == class_idx)

        correct_pixels = torch.sum(pred_class & actual_class, dim=(-2, -1))
        total_pixels = torch.sum(actual_class, dim=(-2, -1))

        if total_pixels > 0:
            accuracy_scores[class_idx] = (correct_pixels / total_pixels).item()

    return accuracy_scores
