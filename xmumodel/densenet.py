from xmumodel.model import Model
from xmuutil.relulayer import ReluLayer
from xmuutil import utils
import tensorflow as tf
import tensorlayer.layers as tl

"""
An implementation of DenseNet used for super-resolution of images as described in:

`Image Super-Resolution Using Dense Skip Connections`
(http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)

"""


class DenseNet(Model):
    def build_model(self, n_dense_blocks=8, scale=8, subpixel=False):
        print("Building DenseNet...")

        norm_input = utils.normalize_color_tf(self.input)
        norm_target = utils.normalize_color_tf(self.target)
        x = tl.InputLayer(norm_input, name='input_layer')

        '''
        extract low level feature
        In Paper <Densely Connected Convolutional Networks>,the filter size here is 7*7
        and followed by a max pool layer
        upscale_input = tl.Conv2d(x,self.feature_size, [7, 7], act = None, name = 'conv0')
        upscale_input = tl.MaxPool2d(upscale_input, [3,3], [2,2], name = 'maxpool0')
        '''
        with tf.variable_scope("low_level_features"):
            x = tl.Conv2d(x, 128, [3, 3], act=None, name='conv0')

        conv1 = x
        with tf.variable_scope("dense_blocks"):
            for i in range(n_dense_blocks):
                x = self.dense_block(x, 16, 8, (3, 3), layer=i)
                x = tl.ConcatLayer([conv1, x], concat_dim=3, name='dense%d/concat_output' % i)

        with tf.variable_scope("bottleneck_layer"):
            '''
            bottleneck layer
            In Paper <Image Super-Resolution Using Dense Skip Connections>
            The channel here is 256
            '''
            x = tl.Conv2d(x, 256, (1, 1), act=None, name='bottleneck')

        with tf.variable_scope("upscale_module"):
            '''
            Paper <Densely Connected Convolutional Networks> using deconv layers to upscale the output
            we provide two methods here: deconv, subpixel
            '''
            if subpixel:
                x = utils.subpixel_upsample(x, 128, scale)
            else:
                x = utils.deconv_upsample(x, 128, (3, 3), scale)

        with tf.variable_scope("reconstruction_layer"):
            output = tl.Conv2d(x, self.n_channels, (3, 3), act=tf.nn.relu, name='reconstruction')

        self.output = tf.clip_by_value(output.outputs, 0.0, 1.0, name="output")
        self.calculate_loss(norm_target, self.output)
        conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=conf)
        self.saver = tf.train.Saver()
        print("Done building!")

    '''
    the implementation of dense block
                    x
                    |\
                    | \
                    |  BN
                    |  relu
                    |  conv2d
                    | /
                    |/
                    x1
                    |
                    [x,x1](concat)

    for a dense block which has n layers,the output is [x,x1,x2....xn]
    while xi mean the output of i-th layers in this dense block
    x: input to pass through the denseblock
    '''
    def dense_block(self, x, growth_rate=16, n_conv=8, kernel_size=(3, 3), layer=0):
        dense_block_output = x
        for i in range(n_conv):
            x = tl.BatchNormLayer(x, name='dense_%d/bn_%d' % (layer, i))
            x = ReluLayer(x, name='dense_%d/relu_%d' % (layer, i))
            x = tl.Conv2d(x, growth_rate, kernel_size, name='dense_%d/conv_%d' % (layer, i))
            # concat the output of layer
            dense_block_output = tl.ConcatLayer([dense_block_output, x], concat_dim=3,
                                                name='dense_%d/concat_%d' % (layer, i))
            x = dense_block_output

        return x