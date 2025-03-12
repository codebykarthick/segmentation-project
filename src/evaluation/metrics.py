import torch


def iou(pred_mask, actual_mask, num_classes=3):
    """
    Compute class-specific IoU using PyTorch operations.

    :param pred_mask: Predicted mask tensor of shape [H, W] or [N, H, W] (batch support)
    :param actual_mask: Ground truth mask tensor of shape [H, W] or [N, H, W] (batch support)
    :param num_classes: Number of segmentation classes (excluding background)
    :return: Mean IoU across classes (excluding background)
    """
    iou_scores = []

    for class_idx in range(1, num_classes):  # Skip background class (0)
        pred_class = (pred_mask == class_idx).to(torch.float32)
        actual_class = (actual_mask == class_idx).to(torch.float32)

        intersection = torch.sum(pred_class * actual_class, dim=(-2, -1))
        union = torch.sum((pred_class + actual_class) > 0, dim=(-2, -1))

        class_iou = intersection / (union + 1e-6)
        iou_scores.append(class_iou)

    return torch.mean(torch.stack(iou_scores)).item() if iou_scores else 0.0


def dice(pred_mask, actual_mask, num_classes=3):
    """
    Compute class-specific Dice Coefficient using PyTorch.

    :param pred_mask: Predicted mask tensor of shape [H, W] or [N, H, W] (batch support)
    :param actual_mask: Ground truth mask tensor of shape [H, W] or [N, H, W] (batch support)
    :param num_classes: Number of segmentation classes (excluding background)
    :return: Mean Dice coefficient across classes (excluding background)
    """
    dice_scores = []

    for class_idx in range(1, num_classes):  # Skip background class (0)
        pred_class = (pred_mask == class_idx).to(torch.float32)
        actual_class = (actual_mask == class_idx).to(torch.float32)

        intersection = torch.sum(pred_class * actual_class, dim=(-2, -1))
        total_pixels = torch.sum(pred_class, dim=(-2, -1)) + \
            torch.sum(actual_class, dim=(-2, -1))

        class_dice = (2 * intersection) / (total_pixels + 1e-6)
        dice_scores.append(class_dice)

    return torch.mean(torch.stack(dice_scores)).item() if dice_scores else 0.0


def pixel_accuracy(pred_mask, actual_mask):
    """
    Compute pixel-wise accuracy between predicted and ground truth masks.

    :param pred_mask: Predicted mask tensor of shape [N, H, W] (batch support)
    :param actual_mask: Ground truth mask tensor of shape [N, H, W] (batch support)
    :return: Mean pixel accuracy across the batch (0 to 1)
    """
    correct_pixels = (pred_mask == actual_mask).sum(
        dim=(-2, -1))  # Per image accuracy
    total_pixels = pred_mask.shape[-2] * \
        pred_mask.shape[-1]  # Pixels per image

    return (correct_pixels / total_pixels).mean().item()  # Mean across batch
