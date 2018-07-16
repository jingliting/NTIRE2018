## NTIRE2018 on image super-resolution
TensorFlow code for [NTIRE2018 challenge](http://www.vision.ee.ethz.ch/en/ntire18/) on image super-resolution.
The challenge has 4 tracks:
##### Track 1: bicubic downscaling x8 competition
##### Track 2: realistic downscaling x4 with mild conditions competition
##### Track 3: realistic downscaling x4 with difficult conditions competition
##### Track 4: wild downscaling x4 competition  
The code uses TensorLayer https://tensorlayer.readthedocs.io/en/stable/. Note that the original experiments were done using torch-autograd, we have so far validated that CIFAR-10 experiments are exactly reproducible in PyTorch, and are in process of doing so for ImageNet (results are very slightly worse in PyTorch, due to hyperparameters).

## Requirements
+ Tensorflow
+ Tensorlayer
+ scipy
+ tqdm
+ argparse

## Downloading datasets
+ Datasets in 2017 for pre-training: [DIV2K2017](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
+ Datasets in 2018: [DIV2K](https://competitions.codalab.org/competitions/18015#learn_the_details)
