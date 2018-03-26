from xmuutil import utils

class DWTData(object):
    def __init__(self, data):
        self.data = data

    def get_test_set(self):
        x_imgs,y_imgs = self.data.get_test_set()
        x_dwt_imgs = utils.get_dwt_images(x_imgs)
        y_dwt_imgs = utils.get_dwt_images(y_imgs)
        return x_dwt_imgs,y_dwt_imgs

    def get_batch(self, batch_size, i):
        x_imgs, y_imgs = self.data.get_batch(batch_size, i)
        x_dwt_imgs = utils.get_dwt_images(x_imgs)
        y_dwt_imgs = utils.get_dwt_images(y_imgs)
        return x_dwt_imgs,y_dwt_imgs







