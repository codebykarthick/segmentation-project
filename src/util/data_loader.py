import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image


train_path = os.path.join(os.getcwd(), "data", "processed", "TrainVal")
train_image_path = os.path.join(train_path, "color")
train_mask_path = os.path.join(train_path, "label")
test_path = os.path.join(os.getcwd(), "data", "processed", "Test")
test_image_path = os.path.join(test_path, "color")
test_mask_path = os.path.join(test_path, "label")

TRAIN_VAL_SPLIT = 0.8
# Change this if you get OOM errors
BATCH_SIZE = 16


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
    def __init__(self, image_dir, mask_dir, transforms=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transforms = transforms

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
        mask = self.convert_mask(mask)

        if self.transforms:
            image = self.transforms(image)

        mask = torch.from_numpy(mask).long()

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


transform = transforms.Compose([
    transforms.ToTensor()
])


def get_seg_data_loaders():
    dataset = SegmentationDataset(train_image_path, train_mask_path, transform)
    train_size = int(TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    test_dataset = SegmentationDataset(
        test_image_path, test_mask_path, transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def get_data_loaders():
    dataset = ImageDataset(train_image_path, transform)
    train_size = int(TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    test_dataset = ImageDataset(test_image_path, transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader
