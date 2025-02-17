import gdown
import os
import shutil
import zipfile

DATASET_FILE_URL="https://drive.google.com/file/d/1CIryq76zXU3ms0Rbpnf_WaL4S7_00fyC/view?usp=share_link"
DATA_PATH=os.path.join(os.getcwd(), "data")
FILE_PATH=os.path.join(DATA_PATH, "cv_dataset.zip")

def setup_data():
    # Check if data folder exists, if yes clean it up.
    if os.path.exists(DATA_PATH):
        print("Cleaning pre-existing data folder")
        shutil.rmtree(DATA_PATH)

    # Create the folder back fresh
    os.mkdir(DATA_PATH)
    print("Downloading dataset")
    # Download the zip file
    gdown.download(DATASET_FILE_URL, FILE_PATH, quiet=False, fuzzy=True)

    # Extract
    print("Zip downloaded! Extracting...")
    with zipfile.ZipFile(FILE_PATH, "r") as z:
        z.extractall("data")
    
    # Clean up the zip file
    print("Zip extracted! Deleting original zip file..")
    os.remove(FILE_PATH)

if __name__ == "__main__":
    setup_data()