import tensorlayer.layers as tl
import tensorflow as tf

class ScaleLayer(tl.Layer):
    def __init__(self,layer = None,scale = 1,name ='scale_layer',
    ):
        # check layer name (fixed)
        tl.Layer.__init__(self, name=name)

        # the input of this layer is the output of previous layer (fixed)
        self.inputs = layer.outputs

        # operation (customized)
        self.outputs = tf.scalar_mul(scale, self.inputs)

        # get stuff from previous layer (fixed)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        # update layer (customized)
        self.all_layers.extend([self.outputs])