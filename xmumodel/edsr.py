from xmumodel.model import Model
import tensorlayer.layers as tl
import tensorflow as tf
from xmuutil import utils
from xmuutil.scalelayer import ScaleLayer
from xmuutil.relulayer import ReluLayer
"""
An implementation of EDSR used for super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""


class EDSR(Model):
    def build_model(self, n_features=256, n_res_blocks=36, scale=8, max_to_keep=100):
        print("Building EDSR...")

        norm_input = utils.normalize_color_tf(self.input)
        norm_target = utils.normalize_color_tf(self.target)
        x = tl.InputLayer(norm_input, name='input_layer')

        # One convolution before res blocks and to convert to required feature depth
        x = tl.Conv2d(x, n_features, (3, 3), name='c')

        # Store the output of the first convolution to add later
        conv_1 = x

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

        with tf.variable_scope("res_blocks"):
            for i in range(n_res_blocks):
                x = self._res_block(x, n_features, (3, 3), scale=scaling_factor, layer=i)
            x = tl.Conv2d(x, n_features, (3, 3), name='res_c')
            x = tl.ElementwiseLayer([conv_1, x], tf.add, name='res_add')

        with tf.variable_scope("upscale_module"):
            x = utils.subpixel_upsample(x, n_features, scale)

        output = tl.Conv2d(x, self.n_channels, (1, 1), act=tf.nn.relu, name='bottleneck')
        self.output = tf.clip_by_value(output.outputs, 0.0, 1.0)

        self.calculate_loss(norm_target, self.output)

        conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=conf)
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        print("Done building!")

    @staticmethod
    def _res_block(x, n_features=64, kernel_size=(3, 3), scale=1.0, layer=0):
        """
        a resBlock is defined in the paper as (excuse the ugly ASCII graph)
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
        """
        nn = tl.Conv2d(x, n_features, kernel_size, act=tf.nn.relu, name='res%d/c1' % layer)
        nn = tl.Conv2d(nn, n_features, kernel_size, act=None, name='res%d/c2' % layer)
        nn = ScaleLayer(nn, scale, name='res%d/scale' % layer)
        n = tl.ElementwiseLayer([x,nn],tf.add, name='res%d/res_add' % layer)
        return n

    @staticmethod
    def _res_block_1(x, n_features=64, kernel_size=(3, 3), scale=1.0, layer=0):
        """
        a resBlock is defined in the paper as (excuse the ugly ASCII graph)
                x
                |\
                | \
                |  relu
                |  conv2d
                |  relu
                |  conv2d
                | /
                |/
                + (addition here)
                |
                result
        """
        nn = ReluLayer(x, name='res%d/ru1' % layer)
        nn = tl.Conv2d(nn, n_features, kernel_size, act=tf.nn.relu, name='res%d/c1' % layer)
        nn = tl.Conv2d(nn, n_features, kernel_size, act=None, name='res%d/c2' % layer)
        nn = ScaleLayer(nn, scale, name='res%d/scale' % layer)
        n = tl.ElementwiseLayer([x, nn], tf.add, name='res%d/res_add' % layer)
        return n

    @staticmethod
    def _res_block_2(x, n_features=64, kernel_size=(3, 3), scale=1.0, layer=0):
        """
        a resBlock is defined in the paper as (excuse the ugly ASCII graph)
                x
                |\
                | \
                |  conv2d
                |  relu
                |  conv2d
                |  relu
                | /
                |/
                + (addition here)
                |
                result
        """
        nn = tl.Conv2d(x, n_features, kernel_size, act=tf.nn.relu, name='res%d/c1' % layer)
        nn = tl.Conv2d(nn, n_features, kernel_size, act=tf.nn.relu, name='res%d/c2' % layer)
        nn = ScaleLayer(nn, scale, name='res%d/scale' % layer)
        n = tl.ElementwiseLayer([x, nn], tf.add, name='res%d/res_add' % layer)
        return n