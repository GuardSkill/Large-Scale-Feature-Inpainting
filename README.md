##   Interactive Fusion Network for High-Resolution Image Inpainting
 [BibTex](#citation)


### Introduction:

<p align='center'>  
  <img src='https://github.com/GuardSkill/Large-Scale-Feature-Inpainting/blob/master/images/Framework.png' width='870'/>
</p>
The framework of Interactive Fusion Network, which is used to replace the generator of the inpainting architecture.

## Prerequisites
- Python 3.6
- PyTorch 1.0 （The version MUST >=1.0）
- NVIDIA GPU + CUDA cuDNN

## Installation
- Clone this repo:
```bash
git clone https://github.com/GuardSkill/Large-Scale-Feature-Inpainting.git
cd Large-Scale-Feature-Inpainting
```
- Install PyTorch and dependencies from http://pytorch.org
- Install python requirements:
```bash
pip install -r requirements.txt
```

## Datasets
### 1) Images
We use [Places2](http://places2.csail.mit.edu), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets. To train a model on the full dataset, download datasets from official websites. 

After downloading, run [`scripts/flist.py`](scripts/flist.py) to generate train, test and validation set file lists. For example, to generate the training set file list on Places2 dataset run:
```bash
mkdir datasets
python ./scripts/flist.py --path path_to_places2_train_set --output ./datasets/places2_train.flist
python ./scripts/flist.py --path path_to_places2_train_set --output ./datasets/places2_edges_val.flist
python ./scripts/flist.py --path path_to_places2_train_set --output ./datasets/masks_test.flist
```

### 2) Irregular Masks
Our model is trained on the irregular mask dataset provided by [Liu et al.](https://arxiv.org/abs/1804.07723). You can download publically available Irregular Mask Dataset from [their website](http://masc.cs.gmu.edu/wiki/partialconv).

Alternatively, you can download [Quick Draw Irregular Mask Dataset](https://github.com/karfly/qd-imd) by Karim Iskakov which is combination of 50 million strokes drawn by human hand.

We additional provide the [code](https://github.com/GuardSkill/AITools/blob/master/image/divide_dataset.py) for dividing the mask maps int to 4 class according to proportion of their corrupted region. (https://arxiv.org/abs/1804.07723)

Please use [`scripts/flist.py`](scripts/flist.py) to generate train, test and validation set masks file lists as explained above.

## Getting Started
Download the pre-trained models using the following links and copy them under `./checkpoints` directory.

[Places2](https://drive.google.com/drive/folders/1KyXz4W4SAvfsGh3NJ7XgdOv5t46o-8aa) | [CelebA](https://drive.google.com/drive/folders/1nkLOhzWL-w2euo0U6amhz7HVzqNC5rqb)

### 1) Training
To train the model, create a `config.yaml` file similar to the [example config file](https://github.com/GuardSkill/Large-Scale-Feature-Inpainting/blob/master/config.yml.example) and copy it under your checkpoints directory. Read the [configuration](#model-configuration) guide for more information on model configuration.

To train the ISNet:
```bash
python train.py --checkpoints [path to checkpoints]
```

### 2) Testing
To test the model, create a `config.yaml` file similar to the [example config file](config.yml.example) and copy it under your checkpoints directory. Read the [configuration](#model-configuration) guide for more information on model configuration.

You can test the model on all three stages: 1) edge model, 2) inpaint model and 3) joint model. In each case, you need to provide an input image (image with a mask) and a grayscale mask file. Please make sure that the mask file covers the entire mask region in the input image. To test the model:
```bash
python test.py \
  --model [stage] \
  --checkpoints [path to checkpoints] \
  --input [path to input directory or file] \
  --mask [path to masks directory or mask file] \
  --output [path to the output directory]
```

We provide some test examples under `./examples` directory. Please download the [pre-trained models](#getting-started) and run:
```bash
python test.py \
  --checkpoints ./checkpoints/places2 
  --input ./examples/places2/images 
  --mask ./examples/places2/masks
  --output ./checkpoints/results
```
This script will inpaint all images in `./examples/places2/images` using their corresponding masks in `./examples/places2/mask` directory and saves the results in `./checkpoints/results` directory. By default `test.py` script is run on stage 3 (`--model=3`).

### 3) Evaluating
To evaluate the model, you need to first run the model in [test mode](#testing) against your validation set and save the results on disk. We provide a utility [`./scripts/metrics.py`](scripts/metrics.py) to evaluate the model using PSNR, SSIM and Mean Absolute Error:

```bash
python ./scripts/metrics.py --data-path [path to validation set] --output-path [path to model output]
```

To measure the Fréchet Inception Distance (FID score) run [`./scripts/fid_score.py`](scripts/fid_score.py). We utilize the PyTorch implementation of FID [from here](https://github.com/mseitzer/pytorch-fid) which uses the pretrained weights from PyTorch's Inception model.

```bash
python ./scripts/fid_score.py --path [path to validation, path to model output] --gpu [GPU id to use]
```


### Model Configuration

The model configuration is stored in a [`config.yaml`](config.yml.example) file under your checkpoints directory. The following tables provide the documentation for all the options available in the configuration file:


## License

Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.

Lots of logic code and readme file comes from [Edge-Connect](https://github.com/knazeri/edge-connect), we sincerely thanks their contribution.


## Citation
If you use this code for your research, please cite our paper <a href="https://">g</a>:

```

```

#### General Model Configurations

Option          | Description
----------------| -----------
MODE            | 1: train, 2: test, 3: eval
MASK            | 1: random block, 2: half, 3: external, 4: external + random block, 5: external + random block + half
SEED            | random number generator seed
GPU             | list of gpu ids, comma separated list e.g. [0,1]
DEBUG           | 0: no debug, 1: debugging mode
VERBOSE         | 0: no verbose, 1: output detailed statistics in the output console

#### Loading Train, Test and Validation Sets Configurations

Option          | Description
----------------| -----------
TRAIN_FLIST     | text file containing training set files list
VAL_FLIST       | text file containing validation set files list
TEST_FLIST      | text file containing test set files list
TRAIN_MASK_FLIST| text file containing training set masks files list (only with MASK=3, 4, 5)
VAL_MASK_FLIST  | text file containing validation set masks files list (only with MASK=3, 4, 5)
TEST_MASK_FLIST | text file containing test set masks files list (only with MASK=3, 4, 5)

#### Training Mode Configurations

Option                 |Default| Description
-----------------------|-------|------------
LR                     | 0.0001| learning rate
D2G_LR                 | 0.1   | discriminator/generator learning rate ratio
BETA1                  | 0.0   | adam optimizer beta1
BETA2                  | 0.9   | adam optimizer beta2
BATCH_SIZE             | 8     | input batch size 
INPUT_SIZE             | 256   | input image size for training. (0 for original size)
MAX_ITERS              | 2e6   | maximum number of iterations to train the model
MAX_STEPS:             | 5000  |maximum number of each epoch
MAX_EPOCHES:           | 100   |maximum number of epoches  100
L1_LOSS_WEIGHT         | 1     | l1 loss weight
FM_LOSS_WEIGHT         | 10    | feature-matching loss weight
STYLE_LOSS_WEIGHT      | 1     | style loss weight
CONTENT_LOSS_WEIGHT    | 1     | perceptual loss weight
INPAINT_ADV_LOSS_WEIGHT| 0.01  | adversarial loss weight
GAN_LOSS               | nsgan | **nsgan**: non-saturating gan, **lsgan**: least squares GAN, **hinge**: hinge loss GAN
GAN_POOL_SIZE          | 0     | fake images pool size
SAVE_INTERVAL          | 1000  | how many iterations to wait before saving model (0: never)
EVAL_INTERVAL          | 0     | how many iterations to wait before evaluating the model (0: never)
SAMPLE_INTERVAL        | 1000  | how many iterations to wait before saving sample (0: never)
SAMPLE_SIZE            | 12    | number of images to sample on each samling interval
EVAL_INTERVAL          | 3     | How many INTERVAL sample while valuation  (0: never  36000 in places)
