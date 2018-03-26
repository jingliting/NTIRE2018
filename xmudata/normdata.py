from xmuutil import utils


class NormalizeData(object):
    def __init__(self, data):
        self.data = data

    def get_batch(self, batch_size, i):
        x_imgs, y_imgs = self.data.get_batch(batch_size, i)
        x_norm_imgs = utils.normalize_color(x_imgs)
        y_norm_imgs = utils.normalize_color(y_imgs)
        return x_norm_imgs, y_norm_imgs

    def get_test_set(self, batch_size):
        x_imgs, y_imgs = self.data.get_test_set(batch_size)
        x_norm_imgs = utils.normalize_color(x_imgs)
        y_norm_imgs = utils.normalize_color(y_imgs)
        return x_norm_imgs,y_norm_imgs


