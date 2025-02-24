# Computer Vision
CV for sem 2 for MSc AI stream. Mini project constitutes for 50% of the final coursework grade.

## Setup
Create a virtual environment (ideally conda) and install the packages listed in the requirements.txt file for the successful execution of the proejct.

## Experiments
Jupyter notebooks that give some extra background information on the design choices made for the scope of the project.

## How to run and other info
### Dataset
The dataset, both original and processed, the project uses is not on the repo for performance reasons and has to be downloaded first. Run `data_setup.py` for the script to download this and unzip in the directory needed by the project files for execution.

If you have access only to the original but not the processed, ensure that it is nested under `data/Dataset/TrainVal` and run `preprocess_augment.py` to generate the needed images for training and testing. It resizes and augments the training data but only resizes the test data for evaluation.