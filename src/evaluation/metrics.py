from collections import defaultdict
import torch
from typing import Dict

# TODO: Review the metrics evaluation logic


def iou(pred_mask: torch.Tensor, actual_mask: torch.Tensor, num_classes: int = 3) -> Dict[int, float]:
    iou_scores = {}

    for class_idx in range(num_classes):
        pred_class = (pred_mask == class_idx)
        actual_class = (actual_mask == class_idx)

        intersection = torch.sum(pred_class & actual_class).item()
        union = torch.sum(pred_class | actual_class).item()

        if union == 0:
            # skip class if both pred and gt are empty (undefined IoU)
            continue
        iou_scores[class_idx] = intersection / (union + 1e-6)

    return iou_scores


def dice(pred_mask: torch.Tensor, actual_mask: torch.Tensor, num_classes: int = 3) -> Dict[int, float]:
    dice_scores = {}

    for class_idx in range(num_classes):
        pred_class = (pred_mask == class_idx).to(torch.float32)
        actual_class = (actual_mask == class_idx).to(torch.float32)

        intersection = torch.sum(pred_class * actual_class).item()
        total_pixels = torch.sum(pred_class).item() + \
            torch.sum(actual_class).item()

        if total_pixels == 0:
            # skip class if both prediction and ground truth are empty (undefined)
            continue
        dice_scores[class_idx] = (2 * intersection) / (total_pixels + 1e-6)

    return dice_scores


def pixel_accuracy(pred_mask: torch.Tensor, actual_mask: torch.Tensor, num_classes: int = 3) -> Dict[int, float]:
    accuracy_scores = {}

    for class_idx in range(num_classes):
        pred_class = (pred_mask == class_idx)
        actual_class = (actual_mask == class_idx)

        correct_pixels = torch.sum(pred_class & actual_class).item()
        total_pixels = torch.sum(actual_class).item()

        if total_pixels == 0:
            pred_pixels = torch.sum(pred_class).item()
            if pred_pixels == 0:
                continue  # skip if both GT and pred are empty
            else:
                accuracy_scores[class_idx] = 0.0  # false positive
        else:
            accuracy_scores[class_idx] = correct_pixels / total_pixels

    return accuracy_scores
