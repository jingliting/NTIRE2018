import tensorlayer.layers as tl
import tensorflow as tf

class TransposedConv2dLayer(tl.Layer):
    def __init__(self,layer = None,out_channels=64, kernel_shape=(3,3), strides=(2,2), padding='SAME', name ='TransposedConv2dLayer',
    ):
        # check layer name (fixed)
        tl.Layer.__init__(self, name=name)

        # the input of this layer is the output of previous layer (fixed)
        self.inputs = layer.outputs

        # operation (customized)
        weights_initializer = tf.truncated_normal_initializer(stddev=0.1)
        self.outputs = tf.contrib.layers.convolution2d_transpose(self.inputs, out_channels, kernel_shape, strides,
                                                         padding=padding, weights_initializer=weights_initializer,
                                                         biases_initializer=tf.constant_initializer(0.0))

        # get stuff from previous layer (fixed)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        # update layer (customized)
        self.all_layers.extend([self.outputs])