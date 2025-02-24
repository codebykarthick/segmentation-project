import os
from PIL import Image
import shutil
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from util import logger

log = logger.setup_logger()

# Base paths
base_path = os.path.join(os.getcwd(), "data")
original_path = os.path.join(base_path, "Dataset")
processed_path = os.path.join(base_path, "processed")

# Train/Test base paths
original_train_path = os.path.join(original_path, "TrainVal")
original_test_path = os.path.join(original_path, "Test")
processed_train_path = os.path.join(processed_path, "TrainVal")
processed_test_path = os.path.join(processed_path, "Test")


def get_image_mask_paths(base):
    """Helper function to get image and mask paths."""
    return os.path.join(base, "color"), os.path.join(base, "label")


# Assign paths using helper function
original_train_img_path, original_train_mask_path = get_image_mask_paths(
    original_train_path)
original_test_img_path, original_test_mask_path = get_image_mask_paths(
    original_test_path)
processed_train_img_path, processed_train_mask_path = get_image_mask_paths(
    processed_train_path)
processed_test_img_path, processed_test_mask_path = get_image_mask_paths(
    processed_test_path)

"""
Refer notebook experiment 1, that explores our dataset train and test images and masks.
We found that 560x600 covers 99%ile of images. We will resize all images to this size.
"""
image_size = (560, 600)

log.info(f"Image size after processing (HxW): {image_size}")


def process(
        image_folder, mask_folder, output_img_folder, output_mask_folder,
        augmentation_transforms, is_test=False):
    """Actually processes the images and masks. Resizes and augments if not test, 
    only resizes if test."""
    for i, file_name in enumerate(os.listdir(image_folder), 1):
        log.info(
            f"Processing image {i}/{len(os.listdir(image_folder))}: {file_name}: {(i/len(os.listdir(image_folder))*100):.2f}%")
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(image_folder, file_name)
            mask_name = file_name.replace(".jpg", ".png")
            mask_path = os.path.join(mask_folder, mask_name)

            # Open images
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")

            # Resize both image and mask together
            image = F.resize(
                image, image_size, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = F.resize(
                mask, image_size, interpolation=transforms.InterpolationMode.NEAREST)

            # Save resized images
            image.save(os.path.join(
                output_img_folder, f"resized_{file_name}"), format="JPEG")
            mask.save(os.path.join(
                output_mask_folder, f"resized_{mask_name}"), format="PNG")

            # Random seed for consistency
            seed = torch.randint(0, 10000, (1,)).item()
            torch.manual_seed(seed)

            if not is_test:
                # Augment images
                for i, transform in enumerate(augmentation_transforms):
                    # Get the same crops for image and mask
                    if isinstance(transform.transforms[0], transforms.RandomRotation):
                        # Get same random rotation angle
                        angle = transforms.RandomRotation.get_params([-30, 30])
                        augmented_image = F.rotate(image, angle)
                        augmented_mask = F.rotate(mask, angle)
                    elif isinstance(transform.transforms[0], transforms.ColorJitter):
                        # Only apply color jitter to image
                        augmented_image = transform(image)
                        augmented_mask = mask
                    else:
                        augmented_image = transform(image)
                        augmented_mask = transform(mask)

                    augmented_image.save(os.path.join(
                        output_img_folder, f"augmented_{i}_{file_name}"))
                    augmented_mask.save(os.path.join(
                        output_mask_folder, f"augmented_{i}_{mask_name}"))
    log.info(
        f"Processed {len(os.listdir(image_folder))} images in {image_folder}")


def preprocess_and_augment():
    """Question 1 solution, process, resize and augment the dataset."""
    if os.path.exists(processed_path):
        log.info("Cleaning old processed images folder.")
        shutil.rmtree(processed_path)

    log.info("Creating processed images folder.")
    os.makedirs(processed_train_img_path, exist_ok=True)
    os.makedirs(processed_train_mask_path, exist_ok=True)
    os.makedirs(processed_test_img_path, exist_ok=True)
    os.makedirs(processed_test_mask_path, exist_ok=True)

    # Now separate augmentations to increase the dataset size
    augment_transforms = [
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),  # Flip
        ]),
        transforms.Compose([
            transforms.RandomRotation(degrees=30),  # Rotate
        ]),
        transforms.Compose([
            transforms.RandomResizedCrop(
                image_size, scale=(0.8, 1.0)),  # Random Crop
        ]),
        transforms.Compose([
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color Jitter
        ]),
    ]

    # Process training & test sets
    process(original_train_img_path, original_train_mask_path, processed_train_img_path,
            processed_train_mask_path, augment_transforms, False)
    process(original_test_img_path, original_test_mask_path, processed_test_img_path,
            processed_test_mask_path, augment_transforms, True)

    log.info("Dataset preprocessing and augmentation completed!")


if __name__ == "__main__":
    log.info("Starting dataset preprocessing and augmentation.")
    preprocess_and_augment()
