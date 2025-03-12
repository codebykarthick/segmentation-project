import torch


def iou(pred_mask, actual_mask):
    """
    Compute IoU efficiently using PyTorch operations.

    :param pred_mask: Predicted mask tensor of shape [H, W] or [N, H, W] (batch support)
    :param gt_mask: Ground truth mask tensor of shape [H, W] or [N, H, W] (batch support)
    :return: IoU score (scalar or per-image tensor)
    """
    pred_mask = (pred_mask > 0).to(
        torch.float32)  # Convert foreground to 1, background to 0
    # Convert ground truth to 1, background to 0
    actual_mask = (actual_mask > 0).to(torch.float32)

    intersection = torch.sum(
        pred_mask * actual_mask, dim=(-2, -1))  # Sum over H, W
    union = torch.sum((pred_mask + actual_mask) > 0, dim=(-2, -1)
                      )  # Sum where at least one is 1

    iou = intersection / (union + 1e-6)  # Avoid division by zero

    return iou.mean().item() if iou.ndim > 0 else iou.item()


def calculate_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute Dice Coefficient using PyTorch (vectorized).

    :param pred_mask: Predicted mask tensor of shape [H, W] or [N, H, W] (batch support)
    :param gt_mask: Ground truth mask tensor of shape [H, W] or [N, H, W] (batch support)
    :return: Dice score (scalar or per-image tensor)
    """
    pred_mask = (pred_mask > 0).to(
        torch.float32)  # Convert to binary (1: object, 0: background)
    gt_mask = (gt_mask > 0).to(torch.float32)

    intersection = torch.sum(
        pred_mask * gt_mask, dim=(-2, -1))  # Sum over H, W
    total_pixels = torch.sum(pred_mask, dim=(-2, -1)) + \
        torch.sum(gt_mask, dim=(-2, -1))  # Sum both masks

    dice = (2 * intersection) / (total_pixels + 1e-6)  # Avoid division by zero

    # Return single scalar for single image
    return dice.mean().item() if dice.ndim > 0 else dice.item()
