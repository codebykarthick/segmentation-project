# Computer Vision
CV for sem 2 for MSc AI stream. Mini project constitutes for 50% of the final coursework grade.

## Setup
Create a virtual environment (ideally conda) and install the packages listed in the requirements.txt file for the successful execution of the project. This assumes your environment has the suitable version of pytorch (version 2.6.0) compiled and installed in your env corresponding to your device (cpu or cuda).

## Miscellaneous
Jupyter notebooks under `experiments/` give some extra background information on the design choices made for the scope of the project.

## Execution
### Downloading dataset:
Dataset contains a large number of images and masks, therefore is not maintained directly on the repository for performance reasons. Therefore to setup dataset run the following

```bash
python3 data_setup.py --data original # Downloads the original dataset
python3 data_setup.py --data processed # Downloads the processed dataset
```

### Augmentation:
If the original dataset is downloaded and it needs to be augmented for the actual training, ensure the original dataset is sucessfully setup under `data/Dataset` in the following format (downloading original from previous step would achieve this).

<pre>
data/
├── Dataset
    ├── TrainVal/
    ├── Test/
</pre>

 and then run

```bash
python3 preprocess_augment.py
```

### Downloading Trained Weights
For testing, the program also allows for downloading of weights from training done beforehand for all the architectures. To get it run

```bash
python3 data_setup.py --weights {unet, autoencoder_seg_encoder_fixed, autoencoder_seg_encoder_tuned,   clip_segmentation_frozen, clip_segmentation_finetuned, prompt_segmentation} 
```

Depending on the type of model you need to test.

### Training and Validation
To perform training and validation using training set and then evaluate the performance in CrossEntropy loss using TestSet to get an estimate run the following in the `src/` directory

```bash
python3 runner.py --model_name {unet, autoencoder, autoencoder_segmentation, clip_segmentation, segment_anything} --mode {train, test} --epochs 10 --batch_size 8 --learning_rate 1e-3
```

Use `-h` for more information.

### Robustness and Metrics
To calculate the IoU, Dice, Pixel accuracy of the saved model or evaluate dice score on various perturbations of an already trained model, run the following in the `src/` directory.

```bash
python3 evaluation_runner.py --model_name {unet, autoencoder_segmentation, clip_segmentation, segment_anything} --eval_methods metrics_iou metrics_dice metrics_pixel-accuracy contrast_inc contrast_dec occlusion gaussian_noise s&p gaussian_blur brightness_inc brightness_dec
```

And the corresponding processes are run and the results are stored in a new json file under `results/` directory.

### UI
To start the UI to test models and see its output, run.

```bash
python3 -m ui.app
```

from the `src/` folder. Which will then start the UI, which has the control buttons to select the model.