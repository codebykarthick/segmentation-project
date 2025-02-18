import os
from PIL import Image
import shutil
import torch
from torchvision import transforms

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


def find_max_image_size(image_folder, max_width=0, max_height=0):
    """Finds the largest common width and height across all images."""
    for file_name in os.listdir(image_folder):
        if file_name.endswith((".jpg", ".png")):
            image_path = os.path.join(image_folder, file_name)
            with Image.open(image_path) as img:
                width, height = img.size
                max_width = max(max_width, width)
                max_height = max(max_height, height)

    return max_width, max_height


# Find the best image_size for our training and testing
max_width, max_height = find_max_image_size(original_train_img_path, 0, 0)
max_width, max_height = find_max_image_size(
    original_test_img_path, max_width, max_height)
image_size = (max_width, max_height)

print(f"Image size after processing: {image_size}")


def process(
        image_folder, mask_folder, output_img_folder, output_mask_folder,
        base_transform, augmentation_transforms, is_test=False):
    """Actually processes the images and masks. Resizes and augments if not test, 
    only resizes if test."""
    for file_name in os.listdir(image_folder):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(image_folder, file_name)
            mask_path = os.path.join(mask_folder, file_name)

            # Open images
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert(
                "L")  # Convert mask to grayscale

            # Resize & Save Original Images
            resized_image = base_transform(image)
            resized_mask = base_transform(mask)

            resized_image_pil = transforms.ToPILImage()(resized_image)
            resized_mask_pil = transforms.ToPILImage()(resized_mask)

            resized_image_pil.save(os.path.join(
                output_img_folder, f"resized_{file_name}"))
            resized_mask_pil.save(os.path.join(
                output_mask_folder, f"resized_{file_name}"))

            if not is_test:
                # Augment images
                for i, transform in enumerate(augmentation_transforms):
                    # Random seed for consistency
                    seed = torch.randint(0, 10000, (1,)).item()
                    torch.manual_seed(seed)
                    augmented_image = transform(image)

                    torch.manual_seed(seed)
                    augmented_mask = transform(mask)

                    augmented_image_pil = transforms.ToPILImage()(augmented_image)
                    augmented_mask_pil = transforms.ToPILImage()(augmented_mask)

                    augmented_image_pil.save(os.path.join(
                        output_img_folder, f"augmented_{i}_{file_name}"))
                    augmented_mask_pil.save(os.path.join(
                        output_mask_folder, f"augmented_{i}_{file_name}"))
    print(
        f"Processed {len(os.listdir(image_folder))} images in {image_folder}")


def preprocess_and_augment():
    """Question 1 solution, process, resize and augment the dataset."""
    if os.path.exists(processed_path):
        print("Cleaning old processed images folder.")
        shutil.rmtree(processed_path)

    print("Creating processed images folder.")
    os.makedirs(processed_train_img_path, exist_ok=True)
    os.makedirs(processed_train_mask_path, exist_ok=True)
    os.makedirs(processed_test_img_path, exist_ok=True)
    os.makedirs(processed_test_mask_path, exist_ok=True)

    # We should resize all the original images first.
    base_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # Now separate augmentations to increase the dataset size
    augment_transforms = [
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=1.0),  # Flip
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomRotation(degrees=30),  # Rotate
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(
                image_size, scale=(0.8, 1.0)),  # Random Crop
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color Jitter
            transforms.ToTensor(),
        ]),
    ]

    # Process training & test sets
    process(original_train_img_path, original_train_mask_path, processed_train_img_path,
            processed_train_mask_path, base_transforms, augment_transforms, False)
    process(original_test_img_path, original_test_mask_path, processed_test_img_path,
            processed_test_mask_path, base_transforms, augment_transforms, True)

    print("Dataset preprocessing and augmentation completed!")


if __name__ == "__main__":
    preprocess_and_augment()
