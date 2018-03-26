from xmumodel.model import Model
import tensorlayer.layers as tl
import tensorflow as tf
from xmuutil.scalelayer import ScaleLayer
from xmuutil import utils

'''
in the paper 
feature_size =64
num_layers = 16
'''
class SRResNet(Model):
    def build_model(self):
        print("Building SRResNet...")

        #input layer
        x=tl.InputLayer(self.input, name='inputlayer')

        x = tl.Conv2d(x, self.feature_size, [3, 3], name='c1')
        conv_1 = x

        # B residual blocks
        for i in range(self.num_layers):
            x = self.__resBlock(x, self.feature_size, layer=i)
        x = tl.Conv2d(x, self.feature_size, [3, 3], name='c2')
        x=tl.BatchNormLayer(x)
        x= tl.ElementwiseLayer([x, conv_1], tf.add, name='res%d/res_add' % (self.num_layers+1))
        # B residual blacks end

        x = utils.subpixel_upsample(x, self.feature_size, self.scale)

        # One final convolution on the upsampling output
        output = tl.Conv2d(x,self.output_channels,[1,1],act=tf.nn.tanh, name='lastLayer')
        # output = tl.Conv2d(x, self.output_channels, [1, 1], act=tf.nn.relu, name='lastLayer')

        self.calculate_loss(output)

        # Tensorflow graph setup... session, saver, etc.
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        print("Done building!")

        """
        Creates a convolutional residual block
        as defined in the paper. More on
        this inside model.py
    
        a resBlock is defined in the paper as
        (excuse the ugly ASCII graph)
            x
            |\
            | \
            |  conv2d
            |  BN
            |  conv2d
            |  BN
            | /
            |/
            + (addition here)
            |
            result
    
        x: input to pass through the residual block
        channels: number of channels to compute
        stride: convolution stride
    
        """

    def __resBlock(self, x, channels=64, kernel_size=[3, 3], scale=1, layer=0):
        nn = tl.Conv2d(x, channels, kernel_size, act=None, name='res%d/c1' % (layer))
        nn=tl.BatchNormLayer(nn,act=tf.nn.relu, name='res%d/b1' % (layer))
        nn = tl.Conv2d(nn, channels, kernel_size, act=None, name='res%d/c2' % (layer))
        nn=tl.BatchNormLayer(nn,act=tf.nn.relu, name='res%d/b2' % (layer))
        nn = ScaleLayer(nn, scale, name='res%d/scale' % (layer))
        n = tl.ElementwiseLayer([x, nn], tf.add, name='res%d/res_add' % (layer))
        return n
