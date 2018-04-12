from xmumodel.model import Model
import tensorlayer.layers as tl
import tensorflow as tf
from xmuutil.scalelayer import ScaleLayer
from xmuutil.relulayer import ReluLayer
from xmuutil import utils
from tqdm import tqdm
import shutil
import os
"""
An implementation of EDSR used for
super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""


class CycleSR(Model):
    def build_model(self, n_features=256, n_res_blocks=36, scale=8, max_to_keep=100):
        print("Building CycleSR...")

        norm_input = utils.normalize_color_tf(self.input)
        norm_target = utils.normalize_color_tf(self.target)
        x = tl.InputLayer(norm_input, name='input_layer')

        # One convolution before res blocks and to convert to required feature depth
        x = tl.Conv2d(x, n_features, (3, 3), name='c')

        # Store the output of the first convolution to add later
        conv_1 = x

        scaling_factor = 0.1

        with tf.variable_scope("res_blocks"):
            for i in range(n_res_blocks):
                x = self._res_block(x, n_features, (3, 3), scale=scaling_factor, layer=i)
            x = tl.Conv2d(x, n_features, (3, 3), name='res_c')
            x = tl.ElementwiseLayer([conv_1, x], tf.add, name='res_add')

        with tf.variable_scope("upscale_module"):
            x = utils.deconv_upsample(x, n_features, kernel=(5, 5), scale=scale)

        output = tl.Conv2d(x, self.n_channels, (1, 1), act=tf.nn.relu, name='bottleneck')
        self.output = tf.clip_by_value(output.outputs, 0.0, 1.0, name='output')
        self.cycle_x = tf.image.resize_bicubic(self.output*255.0+0.5, [48, 48])

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

    def train(self, batch_size=16, iterations=1000, test_every=500, lr_init=1e-4, lr_decay=0.5, decay_every=2e5,
              save_dir="saved_models", reuse=False, reuse_dir=None, reuse_step=1, log_dir="log"):
        """
        Train the neural network
        """
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.mkdir(log_dir)
        os.mkdir(save_dir)

        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(lr_init, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_v)
        self.train_op = optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()
        print("Begin training...")
        with self.sess as sess:
            sess.run(init)
            sess.run(tf.assign(lr_v, lr_init))
            if reuse:
                self.resume(reuse_dir, reuse_step)

            train_writer = tf.summary.FileWriter(log_dir+"/train", sess.graph)
            test_writer = tf.summary.FileWriter(log_dir+"/test", sess.graph)

            test_x, test_y = self.data.get_test_set(batch_size)
            test_feed = {self.input: test_x, self.target: test_y}

            for i in tqdm(range(iterations)):
                if i != 0 and (i % decay_every == 0):
                    new_lr_decay = lr_decay ** (i // decay_every)
                    sess.run(tf.assign(lr_v, lr_init * new_lr_decay))

                x, y = self.data.get_batch(batch_size)
                feed = {
                    self.input: x,
                    self.target: y
                }
                _, cycle_x = sess.run([self.train_op, self.cycle_x], feed)
                cycle_feed = {
                    self.input: cycle_x,
                    self.target: y
                }
                summary, _ = sess.run([self.merged, self.train_op], cycle_feed)
                train_writer.add_summary(summary, i)

                if (i % test_every == 0) or (i == iterations):
                    t_summary = sess.run(self.merged, test_feed)
                    test_writer.add_summary(t_summary, i)
                    self.save(save_dir, i)

            self.save(save_dir, i)
            test_writer.close()
            train_writer.close()