import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from util.logger import setup_logger

log = setup_logger()


train_path = os.path.join(os.getcwd(), "data", "processed", "TrainVal")
train_image_path = os.path.join(train_path, "color")
train_mask_path = os.path.join(train_path, "label")
test_path = os.path.join(os.getcwd(), "data", "processed", "Test")
test_image_path = os.path.join(test_path, "color")
test_mask_path = os.path.join(test_path, "label")

TRAIN_VAL_SPLIT = 0.8


class ImageDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        super().__init__()
        self.image_dir = image_dir
        self.images = sorted(os.listdir(image_dir))
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = Image.open(img_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, defined_transforms=None, prompt_mode=False):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transforms = defined_transforms
        self.prompt_mode = prompt_mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        # Cause mask ends with png
        mask_path = os.path.join(
            self.mask_dir, self.images[index].replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        # Color of Mask classifies if its cat or dog
        mask = Image.open(mask_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        mask = self.convert_mask(mask)
        mask = torch.from_numpy(mask).long()

        if self.prompt_mode:
            foreground_points = torch.nonzero(
                (mask == 1) | (mask == 2), as_tuple=False)
            background_points = torch.nonzero(mask == 0, as_tuple=False)

            sampled_points = []

            if len(foreground_points) > 0:
                num_fg_samples = int(0.3 * len(foreground_points))
                fg_idx = torch.randperm(len(foreground_points))[
                    :num_fg_samples]
                sampled_points.append(foreground_points[fg_idx])

            if len(background_points) > 0:
                num_bg_samples = int(0.1 * len(background_points))
                bg_idx = torch.randperm(len(background_points))[
                    :num_bg_samples]
                sampled_points.append(background_points[bg_idx])

            if sampled_points:
                sampled_points = torch.cat(sampled_points, dim=0)
            else:
                sampled_points = torch.empty((0, 2), dtype=torch.long)

            prompt_mask = torch.zeros(
                (1, mask.shape[0], mask.shape[1]), dtype=torch.float32)
            filtered_mask = torch.zeros_like(mask)

            if len(sampled_points) > 0:
                ys, xs = sampled_points[:, 0], sampled_points[:, 1]
                prompt_mask[0, ys, xs] = 1.0
                filtered_mask[ys, xs] = mask[ys, xs]

            image = torch.cat([image, prompt_mask], dim=0)
            mask = filtered_mask

        return image, mask

    def convert_mask(self, mask):
        """Convert mask to label
        0: Background
        1: Cat
        2: Dog
        """
        mask = np.array(mask)
        label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

        # Extract color channels
        red, green, blue = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]

        # Assign class based on dominant color
        label_mask[(red > blue) & (red > green)] = 1  # Red is highest → Cat
        label_mask[(green > red) & (green > blue) & (
            green > 50)] = 2  # Green is highest → Dog
        # Black/white → Background
        label_mask[(red == blue) & (red == green)] = 0

        return label_mask

    def get_class_label(self, mask_path):
        mask = Image.open(mask_path).convert("RGB")
        label_mask = self.convert_mask(mask)
        if 1 in label_mask:
            return 1  # Cat
        elif 2 in label_mask:
            return 2  # Dog
        return 0  # Background or neither


transform = transforms.Compose([
    transforms.ToTensor()
])


def get_seg_data_loaders(batch_size: int = 8, prompt_mode: bool = False):
    """
    Function that creates the data loaders for the segmentation task.
    Args:
        batch_size (default 8): Size of each batch
        prompt_mode (default False): If set to true, prompt is automatically sampled randomly
        to simulate user prompted points.
    Returns:
        The training, validation and the test set loader.
    """
    log.info(
        f"Creating segmentation loaders with a batch size of: {batch_size}")
    dataset = SegmentationDataset(
        train_image_path, train_mask_path, transform, prompt_mode=prompt_mode)
    train_size = int(TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    test_dataset = SegmentationDataset(
        test_image_path, test_mask_path, transform, prompt_mode=prompt_mode)
    num_workers = min(4, os.cpu_count() // 2)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_data_loaders(batch_size: int = 8):
    """
    Function that creates the data loaders for the segmentation task.
    Args:
        batch_size (default 8): Size of each batch
    Returns:
        The training, validation and the test set loader.
    """
    log.info(
        f"Creating autoencoder loaders with a batch size of: {batch_size}")
    dataset = ImageDataset(train_image_path, transform)
    train_size = int(TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    test_dataset = ImageDataset(test_image_path, transform)
    num_workers = min(4, os.cpu_count() // 2)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
