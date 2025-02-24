# Computer Vision
CV for sem 2 for MSc AI stream. Mini project constitutes for 50% of the final coursework grade.

## Setup
Create a virtual environment (ideally conda) and install the packages listed in the requirements.txt file for the successful execution of the proejct.

## Miscellaneous
Jupyter notebooks under `experiments/` give some extra background information on the design choices made for the scope of the project.

## Execution
### Downloading dataset:
Dataset contains a large number of images and masks, therefore is not maintained directly on the repository for performance reasons. Therefore to setup dataset run the following

```bash
python3 data_setup.py original # Downloads the original dataset
python3 data_setup.py processed # Downloads the processed dataset
```

### Augmentation:
If the original dataset is downloaded and it needs to be augmented for better training, ensure the original dataset is sucessfully setup under `data/Dataset` and run

```bash
python3 preprocess_augment.py
```