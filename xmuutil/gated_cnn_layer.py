import tensorlayer.layers as tl
import tensorflow as tf
import numpy as np


class GatedCNNLayer(tl.Layer):
    def __init__(self, layer=None,kernel_shape=None, name='gated_cnn_layer'):
        """
        Gated PixelCNN layer for the Prior network
        :param layer: X(layer.outputs) - Input tensor in nhwc format
        :param state: state from previous layer
        :param kernel_shape: height and width of kernel
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

            # left side / state input to layer, name='left_conv'
            left = self._mask_conv(layer.state, 2 * in_channel, kernel_shape, mask_type='c', name='left_conv',reuse=False)
            new_state = self._split_and_gate(left, in_channel)

            # convolution from left side to right side. state -> output, name='middle_conv'
            left_to_right_conv = self._mask_conv(left, 2 * in_channel, [1, 1], name='middle_conv')

            # right side / output, name='right_conv1', name='right_conv2'
            right = self._mask_conv(self.inputs, 2 * in_channel, [1, kernel_w], mask_type='b', name='right_conv1')
            right = right + left_to_right_conv
            new_output = self._split_and_gate(right, in_channel)
            new_output = self._mask_conv(new_output, in_channel, [1, 1], mask_type='b', name='right_conv2')
            new_output = new_output + self.inputs
            self.state = new_state
            self.outputs = new_output

        # get stuff from previous layer (fixed)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        # update layer (customized)
        self.all_layers.extend([self.outputs])

    def _split_and_gate(self, tensor, channels):
        """
        Split tensor into two channels of size channels and put the tensors through a PixelCNN gate
        """
        t1 = tensor[:, :, :, :channels]
        t2 = tensor[:, :, :, channels:]
        t1 = tf.nn.tanh(t1)
        t2 = tf.nn.sigmoid(t2)
        return t1 * t2

    def _mask_conv(self, x, out_channels=None, kernel_shape=None, strides=(1, 1), mask_type=None, name='gate_conv', reuse=False):
        with tf.variable_scope(name) as scope:
            batch_size, height, width, in_channel = x.get_shape().as_list()
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
            bias = tf.get_variable('biases', shape=[out_channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

            output = tf.nn.conv2d(x, weights, [1, strides[0], strides[1], 1], padding="SAME")
            output = tf.nn.bias_add(output, bias)

        return output

