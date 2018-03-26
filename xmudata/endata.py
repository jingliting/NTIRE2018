from xmuutil import utils
import random


class EnhanceData(object):
    def __init__(self, data, is_swirl=False):
        self.data = data
        self.is_swirl = is_swirl

    def get_batch(self, batch_size, i):
        x_imgs, y_imgs = self.data.get_batch(batch_size, i)
        select_rg = random.choice(['0', '90'])
        select_axis = random.choice(['0', '1', '2', '-1'])
        x_en_imgs = utils.enhance_imgs(x_imgs, rotate_rg=int(select_rg), flip_axis=int(select_axis), is_swirl=self.is_swirl)
        y_en_imgs = utils.enhance_imgs(y_imgs, rotate_rg=int(select_rg), flip_axis=int(select_axis))
        return x_en_imgs,y_en_imgs

    def get_test_set(self, batch_size):
        x_imgs, y_imgs = self.data.get_test_set(batch_size)
        return x_imgs,y_imgs




