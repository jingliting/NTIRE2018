import tensorlayer.layers as tl
import tensorflow as tf
import numpy as np


class MaskConvLayer(tl.Layer):
    def __init__(self, layer=None, out_channels=None, kernel_shape=None,
                 strides=[1, 1], mask_type=None, name='mask_conv_layer'):
        """
        Convolutional layer capable of being masked, return 2d convolution layer
        :param layer: X(layer.outputs) - input tensor in nhwc format
        :param scale:
        :param out_channels: number of output channels to use
        :param kernel_shape: height and width of kernel
        :param strides: stride size
        :param mask_type: type of mask to use. Masks using one of the following A/B/vertical stack mask from
                        https://arxiv.org/pdf/1606.05328.pdf
        :param name: name for scoping
        """
        # check layer name (fixed)
        tl.Layer.__init__(self, name=name)

        # the input of this layer is the output of previous layer (fixed)
        self.inputs = layer.outputs

        with tf.variable_scope(name) as scope:
            # operation (customized)
            batch_size, height, width, in_channel = self.inputs.get_shape().as_list()
            kernel_h, kernel_w = kernel_shape

            # center coords of kernel/mask
            center_h = kernel_h // 2
            center_w = kernel_w // 2

            if mask_type:
                # using zeros is easier than ones, because horizontal stack
                mask = np.zeros((kernel_h, kernel_w, in_channel, out_channels), dtype=np.float32)
                # vertical stack only, no horizontal stack
                mask[:center_h, :, :, :] = 1
                if mask_type == 'a':  # no center pixel in mask
                    mask[center_h, :center_w, :, :] = 0
                elif mask_type == 'b':  # center pixel in mask
                    mask[center_h, :center_w + 1, :, :] = 1
            else:
                # no mask
                mask = np.ones((kernel_h, kernel_w, in_channel, out_channels), dtype=np.float32)
            # initialize and mask weights
            weights_shape = [kernel_h, kernel_w, in_channel, out_channels]
            # need to experiment with truncated normal vs xavier glorot
            weights_initializer = tf.truncated_normal_initializer(stddev=0.1)
            weights = tf.get_variable("weights", shape=weights_shape, dtype=tf.float32, initializer=weights_initializer)
            weights = weights * mask
            bias = tf.get_variable('bias', shape=[out_channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

            output = tf.nn.conv2d(self.inputs, weights, [1, strides[0], strides[1], 1], padding="SAME")
            output = tf.nn.bias_add(output, bias)
            self.outputs = output
            self.state = output

        # get stuff from previous layer (fixed)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        # update layer (customized)
        self.all_layers.extend([self.outputs])

