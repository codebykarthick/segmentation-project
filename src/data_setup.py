import gdown
import os
import shutil
import sys
import zipfile

DATASET_FILE_URL = "https://drive.google.com/file/d/1CIryq76zXU3ms0Rbpnf_WaL4S7_00fyC/view?usp=share_link"
# Original resize 3000x1800 something bad
# PROCESSED_URL = "https://drive.google.com/file/d/1eK8tYHxv-jtF1KN-B3M5YHSBdlwILZIz/view?usp=share_link"
# First proper - 256x256 better
# PROCESSED_URL = "https://drive.google.com/file/d/1UAQ8E-YOENu8K-kEkFkEl774hgH8-RiH/view?usp=sharing"
# 512x512 Resize
PROCESSED_URL = "https://drive.google.com/file/d/1sVNF0qNFwxaCtkeaICmzhs6PfiaKrGTZ/view?usp=share_link"
data_path = os.path.join(os.getcwd(), "data")
processed_data_path = os.path.join(os.getcwd(), "data")
original_file_path = os.path.join(data_path, "cv_dataset.zip")
processed_file_path = os.path.join(processed_data_path, "processed.zip")


def setup_data(processed=True):
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


if __name__ == "__main__":
    data = sys.argv[1]

    if data == "original":
        setup_data(False)
    elif data == "processed":
        setup_data(True)
    else:
        print("Invalid argument. Please provide 'original' or 'processed' as argument")
