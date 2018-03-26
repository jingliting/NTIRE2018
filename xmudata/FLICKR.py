from xmudata.data import Data
import tensorlayer as tl
from xmuutil import utils
from xmuutil.exception import LargeSizeException

class Flickr(Data):

    def __init__(self, groundtruth_dir, image_size=96, scale=4, test_per=0.2, interp='bicubic'):
        Data.__init__(self, groundtruth_dir, groundtruth_dir, image_size, scale, test_per)
        self.interp = interp

    def get_image_set(self, image_hr_list):
        y_imgs = []
        x_imgs = []
        # use 10 threads to read files
        imgs_hr = tl.visualize.read_images(image_hr_list, self.groundtruth_dir)
        imgs_lr = self.__downsample_imgs(imgs_hr, interp=self.interp, scale=self.scale)

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
    Get downsampled images from HR images
    imgs_hr: list of HR images with shape [batch_size, height, width, channel]
    interp: Interpolation to use for re-sizing (‘nearest’, ‘lanczos’, ‘bilinear’, ‘bicubic’ or ‘cubic’
    """
    def __downsample_imgs(self, imgs_hr, interp, scale=4):
        imgs_lr = []
        for i in range(len(imgs_hr)):
            h, w, _ = imgs_hr[i].shape
            lr_img = tl.visualize.prepro.imresize(imgs_hr[i], size=[h//scale,w//scale], interp=interp)
            imgs_lr.append(lr_img)
        return imgs_lr





