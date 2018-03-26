import scipy.misc
import random
import os
import tensorlayer as tl
from abc import ABCMeta,abstractmethod
import numpy as np


class Data(object, metaclass=ABCMeta):

    """
    Load train and test respectively
    """
    def __init__(self, train_truth_dir, train_data_dir, test_truth_dir = None, test_data_dir=None, image_size = 96, scale = 4, test_percent=0.2):
        self.train_set = []
        self.test_set = []
        self.train_truth_dir = train_truth_dir
        self.train_data_dir = train_data_dir
        if test_truth_dir!=None and test_data_dir!=None:
            self.test_truth_dir = test_truth_dir
            self.test_data_dir = test_data_dir
        else:
            self.test_truth_dir = self.train_truth_dir
            self.test_data_dir = self.train_data_dir

        self.image_size = image_size
        self.scale = scale

        self.test_index = 0

        train_img_files = os.listdir(self.train_data_dir)

        if test_truth_dir!=None and test_data_dir!=None:
            test_img_files = os.listdir(self.test_data_dir)
            self.test_size = len(test_img_files)
            for i in range(len(train_img_files)):
                self.train_set.append(train_img_files[i])
            for i in range(self.test_size):
                self.test_set.append(test_img_files[i])
        else:
            """
             Load set of images in a directory.
             This will automatically allocate a 
             random 20% of the images as a test set
             """
            self.test_size = int(len(train_img_files) * test_percent)

            test_indices = random.sample(range(len(train_img_files)), self.test_size)
            for i in range(len(train_img_files)):
                if i in test_indices:
                    self.test_set.append(train_img_files[i])
                else:
                    self.train_set.append(train_img_files[i])
        return


    """
    Get test set from the loaded dataset

    size (optional): if this argument is chosen,
    each element of the test set will be cropped
    to the first (size x size) pixels in the image.

    returns the test set of your data
    """
    def get_test_set(self,batch_size):
        if self.test_index<self.test_size and self.test_index + batch_size <= self.test_size:
            index = self.test_index
            self.test_index += batch_size
            return self.get_image_set(self.test_set[index:index + batch_size],self.test_data_dir,self.test_truth_dir)

        else:
            return None,None

    @abstractmethod
    def get_image_set(self,image_list,input_dir,ground_truth_dir):
        pass


    """
    Get a batch of images from the training
    set of images.

    batch_size: size of the batch
    original_size: size for target images
    shrunk_size: size for shrunk images

    returns x,y where:
        -x is the input set of shape [-1,shrunk_size,shrunk_size,channels]
        -y is the target set of shape [-1,original_size,original_size,channels]
    """
    def get_batch(self, batch_size):
        img_indices = random.sample(range(len(self.train_set)),batch_size)
        image_list = [self.train_set[i] for i in img_indices]
        return self.get_image_set(image_list,self.train_data_dir,self.train_truth_dir)
