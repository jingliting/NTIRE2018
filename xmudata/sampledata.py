import os
import numpy as np
import random


class SampleData(object):
    # name_postfix for Track1: x8  Track2: x4m  Track3: x4d  Track4: x4w1(2/3/4)
    def __init__(self, data, sample_file=None, repeat_times=4, psnr_threshold=None):
        self.data = data

        sample_lr_list = []
        pos_lr_list = []
        neg_lr_list = []
        f = open(sample_file, 'r')
        for line in f.readlines()[1:-1]:
            row = line.strip()
            lr_file, psnr = row.split(',')
            if float(psnr) < psnr_threshold:
                neg_lr_list.append(lr_file)
                # sample_lr_list.append(lr_file)
            else:
                pos_lr_list.append(lr_file)

        # sample_lr_list = self.__get_lrimg_list(sample_hr_list)
        # sample_lr_list = np.repeat(sample_lr_list, repeat_times)
        self.data.train_set.extend(sample_lr_list)
        self.train_pos_set = pos_lr_list
        self.train_neg_set = neg_lr_list
        print("Reset train set.")

    def get_batch(self, batch_size):
        pos_img_indices = random.sample(range(len(self.train_pos_set)), batch_size // 2)
        neg_img_indices = random.sample(range(len(self.train_neg_set)), batch_size // 2)
        pos_image_list = [self.train_pos_set[i] for i in pos_img_indices]
        neg_image_list = [self.train_neg_set[i] for i in neg_img_indices]

        pos_image_list.extend(neg_image_list)
        x_imgs, y_imgs = self.data.get_image_set(pos_image_list, self.data.train_data_dir, self.data.train_truth_dir)
        return x_imgs, y_imgs

    def get_test_set(self, batch_size):
        x_imgs, y_imgs = self.data.get_test_set(batch_size)
        return x_imgs,y_imgs
