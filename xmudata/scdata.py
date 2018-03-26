from xmuutil import utils

'''
separate channel data
'''
class SeparateChannelData(object):
    def __init__(self, data):
        self.data = data

    def get_test_set(self, batch_size):
        x_imgs, y_imgs = self.data.get_test_set(batch_size)
        x_separate_channel_imgs = utils.split_to_separate_channel(x_imgs)
        y_separate_channel_imgs = utils.split_to_separate_channel(y_imgs)
        return x_separate_channel_imgs, y_separate_channel_imgs

    def get_batch(self, batch_size, i):
        x_imgs, y_imgs = self.data.get_batch(batch_size, i)
        x_separate_channel_imgs = utils.split_to_separate_channel(x_imgs)
        y_separate_channel_imgs = utils.split_to_separate_channel(y_imgs)
        return x_separate_channel_imgs, y_separate_channel_imgs

