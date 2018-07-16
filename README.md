## NTIRE2018 on image super-resolution
TensorFlow code for [NTIRE2018 challenge](http://www.vision.ee.ethz.ch/en/ntire18/) on image super-resolution.
The challenge has 4 tracks:
##### Track 1: bicubic downscaling x8 competition
##### Track 2: realistic downscaling x4 with mild conditions competition
##### Track 3: realistic downscaling x4 with difficult conditions competition
##### Track 4: wild downscaling x4 competition  
The code uses TensorLayer https://tensorlayer.readthedocs.io/en/stable/. Note that the model is based on [EDSR](https://arxiv.org/pdf/1707.02921.pdf) (Enhanced Deep Residual Networks for Single Image Super-Resolution), the championaion of NTIRE2017. We optimized the original EDSR model from three aspects: data augmentation, redesigning up-sampling module and adjusting residual block to tackle track1. Inspired from CycleGAN, we proposed a novel network structure: CycleSR for track2, 3, 4 since the down-sampling methods were unknown.

## Requirements
+ python3
+ Tensorflow
+ Tensorlayer
+ scipy
+ tqdm
+ argparse

## Downloading datasets
Download a dataset of images.       
Place all the images from that dataset into a directory under this one.    
+ Datasets in 2017 for pre-training: [DIV2K2017](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
+ Datasets in 2018: [DIV2K](https://competitions.codalab.org/competitions/18015#learn_the_details)

## Training
The model is designed to support the input images of any size both in training and evaluating stage. Thus you needn't worry about image size. To train, see train.py file for specific parameter definition and run python3 train.py.    
In order to view stats during training, simply run tensorboard --logdir your_train_log_directory.   
The trained model will be saved in the directory you passed in.    
