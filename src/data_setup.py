import gdown
import os
import shutil
import zipfile
import argparse

DATASET_FILE_URL = "https://drive.google.com/file/d/1CIryq76zXU3ms0Rbpnf_WaL4S7_00fyC/view?usp=share_link"
# Original resize 3000x1800 something bad
# PROCESSED_URL = "https://drive.google.com/file/d/1eK8tYHxv-jtF1KN-B3M5YHSBdlwILZIz/view?usp=share_link"
# First proper - 256x256 better
# PROCESSED_URL = "https://drive.google.com/file/d/1UAQ8E-YOENu8K-kEkFkEl774hgH8-RiH/view?usp=sharing"
# 512x512 Resize
PROCESSED_URL = "https://drive.google.com/file/d/1sVNF0qNFwxaCtkeaICmzhs6PfiaKrGTZ/view?usp=share_link"
WEIGHTS_URL = "https://drive.google.com/file/d/182I64mODfRpapaY5kpGYt6MH4W4kHR76/view?usp=sharing"
data_path = os.path.join(os.getcwd(), "data")
processed_data_path = os.path.join(os.getcwd(), "data")
original_file_path = os.path.join(data_path, "cv_dataset.zip")
processed_file_path = os.path.join(processed_data_path, "processed.zip")
weights_path = os.path.join(os.getcwd(), "weights")
weights_file_path = os.path.join(weights_path, "weights.zip")


def setup_data(processed: bool = True):
    """ Download the original or the processed dataset """
    # Check if data folder exists, if yes clean it up.

    if processed:
        print("Downloading the processed dataset")
        path = processed_data_path
        file_url = PROCESSED_URL
        file_path = processed_file_path
    else:
        print("Downloading the original dataset")
        path = data_path
        file_url = DATASET_FILE_URL
        file_path = original_file_path

    if os.path.exists(path):
        print("Cleaning pre-existing data folder")
        shutil.rmtree(path)

    # Create the folder back fresh
    os.mkdir(path)
    # Download the zip file
    gdown.download(file_url, file_path, quiet=False, fuzzy=True)

    # Extract
    print("Zip downloaded! Extracting...")
    with zipfile.ZipFile(file_path, "r") as z:
        z.extractall("data")

    # Clean up the zip file
    print("Zip extracted! Deleting original zip file..")
    os.remove(file_path)


def setup_weights(weight_type: str = "", clear_cache: bool = True):
    tmp_path = os.path.join(weights_path, "tmp")
    if clear_cache:
        # Clean up the zip file
        os.remove(weights_file_path)
        # Clean up temporary directory
        shutil.rmtree(tmp_path)

    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    if not os.path.exists(weights_file_path):
        gdown.download(WEIGHTS_URL, weights_file_path, quiet=False, fuzzy=True)
        with zipfile.ZipFile(weights_file_path, "r") as z:
            z.extractall(tmp_path)

    # Identify and copy only the required weight folder
    tmp_weights_path = os.path.join(tmp_path, "weights")
    for folder in os.listdir(tmp_weights_path):
        print(folder)
        if folder == weight_type:
            src_path = os.path.join(tmp_weights_path, folder)

            # Determine the normalized folder name
            if "autoencoder" in folder.lower():
                dest_folder = "autoencoder_segmentation"
            elif "clip" in folder.lower():
                dest_folder = "clip_segmentation"
            else:
                dest_folder = folder

            dest_path = os.path.join(weights_path, dest_folder)

            # Remove any existing destination folder
            if os.path.exists(dest_path):
                print(f"Removing existing weight folder: {dest_folder}")
                shutil.rmtree(dest_path)

            # Move and rename
            shutil.move(src_path, dest_path)
            print(f"Copied and renamed {folder} to {dest_folder}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Setup")
    parser.add_argument("--data", type=str, choices=["original", "processed"],
                        help="Choose which dataset to download: 'original' or 'processed'")
    parser.add_argument("--weights", type=str,
                        choices=["unet", "autoencoder_seg_encoder_fixed", "autoencoder_seg_encoder_tuned",
                                 "clip_segmentation_frozen", "clip_segmentation_finetuned", "prompt_segmentation"],
                        help="Specify which model weights to download")
    parser.add_argument("--clear_cache", type=bool, default=False,
                        help="Clear downloaded zip file and download new.")

    args = parser.parse_args()
    data = args.data

    if data == "original":
        setup_data(False)
    elif data == "processed":
        setup_data(True)

    if args.weights:
        print(
            f"Downloading pretrained weights for: {args.weights}")
        setup_weights(args.weights, args.clear_cache)
    else:
        print("Skipping weights download")
