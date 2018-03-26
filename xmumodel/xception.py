from xmumodel.model import Model
import tensorlayer.layers as tl
import tensorflow as tf
from xmuutil import utils
from xmuutil.relulayer import ReluLayer
"""
An implementation of EDSR used for
super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""
class Xception(Model):
    def buildModel(self):
        print("Building Xception...")

        #input layer
        x=tl.InputLayer(self.input, name='inputlayer')

        # ===========ENTRY FLOW==============
        # Block 1
        net = tl.Conv2d(x, 32, [3, 3], padding='SAME', name='block1_conv1')
        net = tl.BatchNormLayer(net, name='block1_bn1')
        net = ReluLayer(net, name='block1_relu1')
        net = tl.Conv2d(net, 64, [3, 3], padding='SAME', name='block1_conv2')
        net = tl.BatchNormLayer(net, name='block1_bn2')
        net = ReluLayer(net, name='block1_relu2')
        residual = tl.Conv2d(net, 128, [1, 1], name='block1_res_conv')
        residual = tl.BatchNormLayer(residual, name='block1_res_bn')

        # Block 2
        net = tl.SeparableConv2dLayer(net, 128, [3, 3], padding='SAME', name='block2_dws_conv1')
        net = tl.BatchNormLayer(net, name='block2_bn1')
        net = ReluLayer(net, name='block2_relu1')
        net = tl.SeparableConv2dLayer(net, 128, [3, 3], padding='SAME',name='block2_dws_conv2')
        net = tl.BatchNormLayer(net, name='block2_bn2')
        # net = tl.MaxPool2d(net, [3, 3],  padding='SAME', name='block2_max_pool')
        net = tl.ElementwiseLayer([net, residual], tf.add, name='block2_add')
        residual = tl.Conv2d(net, 256, [1, 1], name='block2_res_conv')
        residual = tl.BatchNormLayer(residual, name='block2_res_bn')

        # Block 3
        net = ReluLayer(net, name='block3_relu1')
        net = tl.SeparableConv2dLayer(net, 256, [3, 3], padding='SAME',name='block3_dws_conv1')
        net = tl.BatchNormLayer(net, name='block3_bn1')
        net = ReluLayer(net, name='block3_relu2')
        net = tl.SeparableConv2dLayer(net, 256, [3, 3], padding='SAME',name='block3_dws_conv2')
        net = tl.BatchNormLayer(net, name='block3_bn2')
        # net = tl.MaxPool2d(net, [3, 3],  padding='SAME', name='block3_max_pool')
        net = tl.ElementwiseLayer([net, residual], tf.add,name='block3_add')
        residual = tl.Conv2d(net, 728, [1, 1],  name='block3_res_conv')
        residual = tl.BatchNormLayer(residual, name='block3_res_bn')

        # Block 4
        net = ReluLayer(net, name='block4_relu1')
        net = tl.SeparableConv2dLayer(net, 728, [3, 3], padding='SAME',name='block4_dws_conv1')
        net = tl.BatchNormLayer(net, name='block4_bn1')
        net = ReluLayer(net, name='block4_relu2')
        net = tl.SeparableConv2dLayer(net, 728, [3, 3], padding='SAME',name='block4_dws_conv2')
        net = tl.BatchNormLayer(net, name='block4_bn2')
        # net = tl.MaxPool2d(net, [3, 3],  padding='SAME', name='block4_max_pool')
        net = tl.ElementwiseLayer([net, residual],tf.add, name='block4_add')

        # ===========MIDDLE FLOW===============
        for i in range(8):
            block_prefix = 'block%s_' % (str(i + 5))

            residual = net
            net = ReluLayer(net, name=block_prefix + 'relu1')
            net = tl.SeparableConv2dLayer(net, 728, [3, 3], padding='SAME',name=block_prefix + 'dws_conv1')
            net = tl.BatchNormLayer(net, name=block_prefix + 'bn1')
            net = ReluLayer(net, name=block_prefix + 'relu2')
            net = tl.SeparableConv2dLayer(net, 728, [3, 3], padding='SAME',name=block_prefix + 'dws_conv2')
            net = tl.BatchNormLayer(net, name=block_prefix + 'bn2')
            net = ReluLayer(net, name=block_prefix + 'relu3')
            net = tl.SeparableConv2dLayer(net, 728, [3, 3],  padding='SAME',name=block_prefix + 'dws_conv3')
            net = tl.BatchNormLayer(net, name=block_prefix + 'bn3')
            net = tl.ElementwiseLayer([net, residual], tf.add, name=block_prefix + 'add')

        # ========EXIT FLOW============
        residual = tl.Conv2d(net, 1024, [1, 1],  name='block12_res_conv')
        residual = tl.BatchNormLayer(residual, name='block12_res_bn')
        net = ReluLayer(net, name='block13_relu1')
        net = tl.SeparableConv2dLayer(net, 728, [3, 3],  padding='SAME',name='block13_dws_conv1')
        net = tl.BatchNormLayer(net, name='block13_bn1')
        net = ReluLayer(net, name='block13_relu2')
        net = tl.SeparableConv2dLayer(net, 1024, [3, 3],  padding='SAME',name='block13_dws_conv2')
        net = tl.BatchNormLayer(net, name='block13_bn2')
        # net = tl.MaxPool2d(net, [3, 3],  padding='SAME', name='block13_max_pool')
        net = tl.ElementwiseLayer([net, residual],tf.add, name='block13_add')
        
        net = utils.subpixelupsample(net, self.output_channels*2**2, scale=self.scale)
        output = net
        # output = tl.Conv2d(net, self.output_channels, [1, 1], act=tf.nn.relu, name='lastLayer')

        # net = tl.SeparableConv2dLayer(net, 1536, [3, 3], name='block14_dws_conv1')
        # net = tl.BatchNormLayer(net, name='block14_bn1')
        # net = ReluLayer(net, name='block14_relu1')
        # net = tl.SeparableConv2dLayer(net, 2048, [3, 3], name='block14_dws_conv2')
        # net = tl.BatchNormLayer(net, name='block14_bn2')
        # net = ReluLayer(net, name='block14_relu2')
        # 
        # net = slim.avg_pool2d(net, [10, 10], name='block15_avg_pool')
        # # Replace FC layer with conv layer instead
        # net = tl.Conv2d(net, 2048, [1, 1], name='block15_conv1')

        self.output = output.outputs

        self.cacuLoss(output)

        # Tensorflow graph setup... session, saver, etc.
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        print("Done building!")
