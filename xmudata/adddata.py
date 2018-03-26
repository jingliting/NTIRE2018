import random
import numpy as np


class AddData(object):
    # name_postfix for Track1: x8  Track2: x4m  Track3: x4d  Track4: x4w1(2/3/4)
    def __init__(self, data, ratio=0.2):
        self.data = data
        self.ratio = ratio

    def get_batch(self, batch_size):
        img_indices = random.sample(range(len(self.data.train_set)), int(batch_size * (1-self.ratio)))
        valid_img_indices = random.sample(range(len(self.data.test_set)), int(batch_size * self.ratio))
        image_list = [self.data.train_set[i] for i in img_indices]
        valid_image_list = [self.data.test_set[i] for i in valid_img_indices]

        x_imgs, y_imgs = self.data.get_image_set(image_list, self.data.train_data_dir, self.data.train_truth_dir)
        valid_x_imgs, valid_y_imgs = self.data.get_image_set(valid_image_list, self.data.test_data_dir,
                                                             self.data.test_truth_dir)
        return np.concatenate((x_imgs, valid_x_imgs), axis=0), np.concatenate((y_imgs, valid_y_imgs), axis=0)

    def get_test_set(self, batch_size):
        x_imgs, y_imgs = self.data.get_test_set(batch_size)
        return x_imgs,y_imgs
