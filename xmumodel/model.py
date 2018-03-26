import tensorflow as tf
from tqdm import tqdm
from abc import ABCMeta,abstractmethod
import os
import shutil
from xmuutil import utils


class Model(object, metaclass=ABCMeta):
    def __init__(self, n_channels=3):
        self.n_channels = n_channels
        self.input = tf.placeholder(tf.float32, [None, None, None, self.n_channels], name='lr_input')
        self.target = tf.placeholder(tf.float32, [None, None, None, self.n_channels], name='hr_target')
        self.output = None
        self.train_op = None

    @abstractmethod
    def build_model(self):
        pass

    def save(self, save_dir='saved_models', global_step=None):
        """
        Save the current state of the network to file
        """
        print("Saving...")
        # tl.files.save_npz(self.all_params, name=savedir + '/model.npz', sess=self.sess)
        self.saver.save(self.sess, save_dir+"/model", global_step=global_step)
        print("Saved!")

    def resume(self, save_dir='saved_models', global_step=None):
        """
        Resume network from previously saved weights
        global_step: the specific iteration to resume, if global_step is None,resume the latest checkpoint
        """
        if os.path.exists(save_dir):
            if global_step is None:
                checkpoint_path_to_resume = tf.train.latest_checkpoint(save_dir)
            else:
                checkpoint_path_to_resume = save_dir + '/model-' + str(global_step)
            print("Restoring from " + checkpoint_path_to_resume)
            self.saver.restore(self.sess, checkpoint_path_to_resume)
            print("Restored!")

    def calculate_loss(self, target=None, output=None):
        self.loss = tf.reduce_mean(tf.losses.absolute_difference(target, output))
        psnr = utils.psnr_tf(target, output, is_norm=True)

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("PSNR", psnr)

        # # Image summaries for input, target, and output
        # tf.summary.image("input_image", tf.cast(self.input, tf.uint8))
        # tf.summary.image("target_image", tf.cast(self.target, tf.uint8))
        # tf.summary.image("output_image", tf.cast(x.outputs, tf.uint8))
        self.merged = tf.summary.merge_all()

    def set_data(self, data):
        """
        Function to setup your input data pipeline
        """
        self.data = data

    def predict(self, x):
        """
        Estimate the trained model
        x: (tf.float32, [batch_size, h, w, output_channels])
        """
        return self.sess.run(self.output, feed_dict={self.input: x})

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

            test_feed = []
            while True:
                test_x, test_y = self.data.get_test_set(batch_size)
                if test_x is not None and test_y is not None:
                    test_feed.append({
                            self.input: test_x,
                            self.target: test_y
                    })
                else:
                    break

            for i in tqdm(range(iterations)):
                if i != 0 and (i % decay_every == 0):
                    new_lr_decay = lr_decay ** (i // decay_every)
                    sess.run(tf.assign(lr_v, lr_init * new_lr_decay))

                x, y = self.data.get_batch(batch_size)
                feed = {
                    self.input: x,
                    self.target: y
                }
                summary, _ = sess.run([self.merged, self.train_op], feed)
                train_writer.add_summary(summary, i)

                if (i % test_every == 0) or (i == iterations):
                    test_writer.add_summary(self.merged, i)
                    self.save(save_dir, i)

            test_writer.close()
            train_writer.close()