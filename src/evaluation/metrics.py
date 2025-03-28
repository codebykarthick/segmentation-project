from collections import defaultdict
import torch

# TODO: Review the metrics evaluation logic


def iou(pred_mask, actual_mask, num_classes=3):
    """
    Compute class-specific IoU using PyTorch
    :param pred_mask: Predicted mask of shape [H, W]
    :param actual_mask: Ground-truth mask of shape [H, W]
    :param num_classes: Number of segmentation classes (including background=0)
    :return: Mean IoU across all classes that appear in the masks
    """
    iou_scores = {}

    for class_idx in range(num_classes):
        pred_class = (pred_mask == class_idx)
        actual_class = (actual_mask == class_idx)

        intersection = torch.sum(pred_class & actual_class, dim=(-2, -1))
        union = torch.sum(pred_class | actual_class, dim=(-2, -1))

        # If the union is zero, it means neither prediction nor ground truth
        # had any pixels for this class, so skip it.
        non_empty_mask = (actual_class != 0)
        if non_empty_mask.any():
            # Compute IoU only for images (or pixels) where this class actually appears
            class_iou = intersection / (union + 1e-6)
            iou_scores[class_idx] = class_iou.item()

    return iou_scores


def dice(pred_mask, actual_mask, num_classes=3):
    """
    Compute class-specific Dice Coefficient using PyTorch.

    :param pred_mask: Predicted mask tensor of shape [H, W] or [N, H, W] (batch support)
    :param actual_mask: Ground truth mask tensor of shape [H, W] or [N, H, W] (batch support)
    :param num_classes: Number of segmentation classes (excluding background)
    :return: Mean Dice coefficient across classes (excluding background)
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


def pixel_accuracy(pred_mask, actual_mask, num_classes=3):
    """
    Compute pixel-wise accuracy between predicted and ground truth masks.

    :param pred_mask: Predicted mask tensor of shape [N, H, W] (batch support)
    :param actual_mask: Ground truth mask tensor of shape [N, H, W] (batch support)
    :return: Mean pixel accuracy across the batch (0 to 1)
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
