from xmumodel.model import Model
import tensorlayer.layers as tl
import tensorflow as tf
from xmuutil import utils
from xmuutil.scalelayer import ScaleLayer
"""
An implementation of EDSR used for
super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""
class EDSR(Model):
    def buildModel(self):
        print("Building EDSR...")

        #input layer
        x=tl.InputLayer(self.input, name='inputlayer')

        # One convolution before res blocks and to convert to required feature depth
        x = tl.Conv2d(x, self.feature_size, [3, 3], name='c')

        # Store the output of the first convolution to add later
        conv_1 = x

        """
        This creates `num_layers` number of resBlocks

        """

        """
        Doing scaling here as mentioned in the paper:

        `we found that increasing the number of feature
        maps above a certain level would make the training procedure
        numerically unstable. A similar phenomenon was
        reported by Szegedy et al. We resolve this issue by
        adopting the residual scaling with factor 0.1. In each
        residual block, constant scaling layers are placed after the
        last convolution layers. These modules stabilize the training
        procedure greatly when using a large number of filters.
        In the test phase, this layer can be integrated into the previous
        convolution layer for the computational efficiency.'

        """
        scaling_factor = 0.1

        # Add the residual blocks to the model
        for i in range(self.num_layers):
            x = self.__resBlock(x, self.feature_size, scale=scaling_factor, layer=i)

        # One more convolution, and then we add the output of our first conv layer
        x = tl.Conv2d(x, self.feature_size, [3, 3], act = None, name = 'm1')
        x = tl.ElementwiseLayer([conv_1,x],tf.add, name='res_add')

        x = utils.subpixelupsample(x,self.feature_size,self.scale)

        # One final convolution on the upsampling output
        output = tl.Conv2d(x,self.output_channels,[1,1],act=tf.nn.relu, name='lastLayer')
        self.output = tf.clip_by_value(output.outputs, 0.0, 1.0)

        self.cacuLoss(output)

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
        |  relu
        |  conv2d
        | /
        |/
        + (addition here)
        |
        result

    x: input to pass through the residual block
    channels: number of channels to compute
    stride: convolution stride
    
    """
    def __resBlock(self, x, channels = 64, kernel_size = (3, 3), scale = 1.0,layer = 0):
        nn = tl.Conv2d(x, channels, kernel_size, act=tf.nn.relu, name='res%d/c1'%(layer))
        nn = tl.Conv2d(nn, channels, kernel_size, act=None, name='res%d/c2'%(layer))
        nn = ScaleLayer(nn,scale, name='res%d/scale'%(layer))
        n = tl.ElementwiseLayer([x,nn],tf.add, name='res%d/res_add'%(layer))
        return n