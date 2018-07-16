## Code for NTIRE [NTIRE2018 challenge](http://www.vision.ee.ethz.ch/en/ntire18/) on image super-resolution
The challenge has 4 tracks:
##### Track 1: bicubic downscaling x8 competition
##### Track 2: realistic downscaling x4 with mild conditions competition
##### Track 3: realistic downscaling x4 with difficult conditions competition
##### Track 4: wild downscaling x4 competition
The 

## Downloading datasets
+ (Datasets in 2017 for pre-training: [DIV2K2017](https://data.vision.ee.ethz.ch/cvl/DIV2K/))
+ (Datasets in 2018: [DIV2K](https://competitions.codalab.org/competitions/18015#learn_the_details))


## Training
CHECKPOINT_FILE = ${CHECKPOINT_DIR}/inception_v3.ckpt  # Example
$ python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=inception_v3


## Evaluating performance
GROUNDTRUTH_DIR = data/DIV2K/DIV2K_valid_LR_x8 # Example
DATA_DIR = data/DIV2K/DIV2K_valid_HR # Example
CHECKPOINT_DIR = result/ckpt  # Example
$ python3 predict.py \
    --groundtruthdir= \
    --datadir=${DATA_DIR} \
    --postfixlen=2 \
    --reusedir=${CHECKPOINT_DIR} \
    --step=100000 \
    --outdir=out
