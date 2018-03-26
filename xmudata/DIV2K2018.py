from xmudata.data import Data
import tensorlayer as tl
from xmuutil import utils
import os
from xmuutil.exception import LargeSizeException


class DIV2K2018(Data):

    def __init__(self, train_truth_dir, train_data_dir, test_truth_dir = None, test_data_dir=None, image_size = 96, scale = 4, postfix_len = 3, test_per = 0.2):
        Data.__init__(self, train_truth_dir, train_data_dir,test_truth_dir,test_data_dir, image_size, scale, test_per)
        self.postfix_len = postfix_len

    def get_image_set(self, image_lr_list,input_dir,ground_truth_dir):
        y_imgs = []
        x_imgs = []
        # use 10 threads to read files
        imgs_lr = tl.visualize.read_images(image_lr_list, input_dir)
        image_hr_list = self.__get_hrimg_list(image_lr_list)
        imgs_hr = tl.visualize.read_images(image_hr_list, ground_truth_dir)

        for i in range(len(imgs_lr)):
            #crop the image randomly
            try:
                x_img,y_img = utils.crop(imgs_lr[i], imgs_hr[i], self.image_size, self.image_size, self.scale, is_random=True)
            except LargeSizeException as e:
                print(e)
            else:
                y_imgs.append(y_img)
                x_imgs.append(x_img)
        return x_imgs, y_imgs

    """
    get the corresponding low-resolution image names
    :parameter
    image_list: high-resolotion image names
    """
    def __get_hrimg_list(self,image_lr_list):
        image_hr_list= []
        for i in range(len(image_lr_list)):
            name_lr,postfix = os.path.splitext(image_lr_list[i])
            name_hr = name_lr[:-1*self.postfix_len]
            image_hr_list.append(name_hr+postfix)
        return image_hr_list


