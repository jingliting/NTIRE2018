import tensorflow as tf
import tensorlayer.layers as tl
from xmuutil.relulayer import ReluLayer
from xmuutil.scalelayer import ScaleLayer
from xmuutil.transposed_conv2d_layer import TransposedConv2dLayer
from xmuutil import utils
from xmumodel.model import Model
from tqdm import tqdm
import os
import shutil


class CycleSR(Model):
    def __init__(self, n_features=128, n_resblocks=16, n_channels=3):
        self.n_features = n_features
        self.n_resblocks = n_resblocks
        self.n_channels = n_channels

        self.input_lr = tf.placeholder(tf.float32, [None, None, None, n_channels], name='input_A')
        self.input_hr = tf.placeholder(tf.float32, [None, None, None, n_channels], name='input_B')

    def build_model(self, n_channels=3):
        print("Building CycleSR..")

        self.generator()

        self.calculate_loss()

        # Tensorflow graph setup... session, saver, etc.
        conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=conf)
        self.saver = tf.train.Saver(max_to_keep=100)
        print("Done building!")

    def generator(self):

        self.norm_input_lr = utils.normalize_color_tf(self.input_lr)
        self.norm_input_hr = utils.normalize_color_tf(self.input_hr)

        x = tl.InputLayer(self.norm_input_lr, name='a_input')
        x = tl.Conv2d(x, self.n_features, (3, 3), (1, 1), name='a_conv')
        a_conv = x

        scaling_factor = 0.1
        with tf.variable_scope('res_blocks'):
            for i in range(self.n_resblocks):
                x = self.resBlock(x, self.n_features, (3, 3), (1, 1), scale=scaling_factor, name='res_'+str(i))

        x = tl.Conv2d(x, self.n_features, (3, 3), (1, 1), name='a_pos_conv')
        x = tl.ElementwiseLayer([a_conv, x], tf.add, name='a_res_add')

        x = TransposedConv2dLayer(x, self.n_features, (5, 5), (2, 2), name='a_upscale_1')
        x = tl.Conv2d(x, self.n_features, (3, 3), (1, 1), act=tf.nn.relu, name='a_temp_conv')
        x = TransposedConv2dLayer(x, self.n_features, (5, 5), (2, 2), name='a_upscale_2')

        x = tl.Conv2d(x, self.n_channels, (1, 1), (1, 1), act=tf.nn.relu, name='a_bottleneck')
        self.fake_hr = tf.clip_by_value(x.outputs, 0.0, 1.0)

        y = tl.InputLayer(self.norm_input_hr, name='b_input')
        y = tl.Conv2d(y, self.n_features, (1, 1), (1, 1), act=tf.nn.relu, name='b_bottleneck')

        y = tl.Conv2d(y, self.n_features, (5, 5), (2, 2), name='b_downscale_1')
        y = tl.Conv2d(y, self.n_features, (3, 3), (1, 1), act=tf.nn.relu, name='b_temp_conv')
        y = tl.Conv2d(y, self.n_features, (5, 5), (2, 2), name='b_downscale_2')

        b_conv = y
        with tf.variable_scope('res_blocks', reuse=True):
            tl.set_name_reuse(True)
            for i in range(self.n_resblocks):
            # for i in range(self.n_resblocks - 1, -1, -1):
                y = self.resBlock(y, self.n_features, (3, 3), (1, 1), scale=scaling_factor, name='res_'+str(i),
                                  reuse=False)
        y = tl.Conv2d(y, self.n_features, (3, 3), (1, 1), name='b_pos_conv')
        y = tl.ElementwiseLayer([b_conv, y], tf.add, name='b_res_add')

        y = tl.Conv2d(y, self.n_channels, (3, 3), (1, 1), name='b_conv')
        self.fake_lr = tf.clip_by_value(y.outputs, 0.0, 1.0)

    def resBlock(self, x, n_features=64, kernel_size = (3, 3), strides=(1,1), padding='SAME', scale=1.0, name='res',
                 reuse=False):
        nn = ReluLayer(x, name=name+'/ru1')
        nn = tl.Conv2d(nn, n_features, kernel_size, strides, padding=padding, name=name+'/c1')
        nn = ReluLayer(nn, name=name+'/ru2')
        nn = tl.Conv2d(nn, n_features, kernel_size, strides, padding=padding, name=name+'/c2')
        nn = ScaleLayer(nn, scale, name=name+'/scale')
        if reuse:
            n = tl.ElementwiseLayer([x,nn],tf.subtract, name=name+'/add')
        else:
            n = tl.ElementwiseLayer([x, nn], tf.add, name=name + '/add')
        return n

    def calculate_loss(self):
        # downsize_b = tf.image.resize_bicubic(self.norm_input_b, [48, 48])
        self.loss_lr = self.cycle_consistency_loss(self.norm_input_lr, self.fake_lr)
        self.loss_hr = self.cycle_consistency_loss(self.norm_input_hr, self.fake_hr)  # HR loss

        PSNR = utils.psnr_tf(self.norm_input_hr, self.fake_hr, is_norm=True)

        # Scalar to keep track for loss
        summary_loss = tf.summary.scalar("loss_b", self.loss_hr)
        summary_psnr = tf.summary.scalar("PSNR", PSNR)

        streaming_loss, self.streaming_loss_update = tf.contrib.metrics.streaming_mean(self.loss_hr)
        streaming_loss_scalar = tf.summary.scalar('loss',streaming_loss)

        streaming_psnr, self.streaming_psnr_update = tf.contrib.metrics.streaming_mean(PSNR)
        streaming_psnr_scalar = tf.summary.scalar('PSNR',streaming_psnr)

        # # Image summaries for input, target, and output
        # input_image = tf.summary.image("input_image", tf.cast(self.input, tf.uint8))
        # target_image = tf.summary.image("target_image", tf.cast(self.target, tf.uint8))
        # output_image = tf.summary.image("output_image", tf.cast(x.outputs, tf.uint8))

        self.train_merge = tf.summary.merge([summary_loss,summary_psnr])
        self.test_merge = tf.summary.merge([streaming_loss_scalar,streaming_psnr_scalar])

    def train(self,batch_size= 10, iterations=1000, lr_init=1e-4, lr_decay=0.5, decay_every=2e5,
              save_dir="saved_models",reuse=False,reuse_dir=None,reuse_step=None, log_dir="log"):
        #create the save directory if not exist
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.mkdir(log_dir)
        #Make new save directory
        os.mkdir(save_dir)

        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(lr_init, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_v)
        train_lr_op = optimizer.minimize(self.loss_lr)
        train_hr_op = optimizer.minimize(self.loss_hr)

        #Operation to initialize all variables
        init = tf.global_variables_initializer()
        print("Begin training...")
        with self.sess as sess:
            #Initialize all variables
            sess.run(init)
            if reuse:
                self.resume(reuse_dir, global_step=reuse_step)

            #create summary writer for train
            train_writer = tf.summary.FileWriter(log_dir+"/train",sess.graph)

            #If we're using a test set, include another summary writer for that
            test_writer = tf.summary.FileWriter(log_dir+"/test",sess.graph)
            test_feed = []
            while True:
                test_x,test_y = self.data.get_test_set(batch_size)
                if test_x!=None and test_y!=None:
                    test_feed.append({
                            self.input_lr:test_x,
                            self.input_hr:test_y
                    })
                else:
                    break

            sess.run(tf.assign(lr_v, lr_init))
            #This is our training loop
            for i in tqdm(range(iterations)):
                if i != 0 and (i % decay_every == 0):
                    new_lr_decay = lr_decay ** (i // decay_every)
                    sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                x,y = self.data.get_batch(batch_size)
                feed = {
                    self.input_lr:x,
                    self.input_hr:y
                }
                summary, fake_hr, fake_lr, _, _ = sess.run([self.train_merge, self.fake_hr, self.fake_lr,
                                                   train_lr_op, train_hr_op], feed)
                cycle_lr_feed = {
                    self.input_lr: x,
                    self.input_hr: fake_hr
                }
                _ = sess.run([train_lr_op], cycle_lr_feed)

                cycle_hr_feed = {
                    self.input_lr: fake_lr,
                    self.input_hr: y
                }
                _ = sess.run([train_hr_op], cycle_hr_feed)
                #Write train summary for this step
                train_writer.add_summary(summary,i)

                # Test every 500 iterations and save our trained model
                if (i!=0 and i % 10000 == 0) or (i == iterations):
                    sess.run(tf.local_variables_initializer())
                    for j in range(len(test_feed)):
                        # sess.run([self.streaming_loss_update],feed_dict=test_feed[j])
                        sess.run([self.streaming_loss_update, self.streaming_psnr_update], feed_dict=test_feed[j])
                    streaming_summ = sess.run(self.test_merge)
                    # Write test summary
                    test_writer.add_summary(streaming_summ, i)

                    self.save(save_dir, i)

            test_writer.close()
            train_writer.close()

    def predict_a(self, x):
        return self.sess.run(self.fake_lr, feed_dict={self.input_hr: x})

    def predict_b(self, x):
        return self.sess.run(self.fake_hr, feed_dict={self.input_lr: x})

    def save(self, savedir='saved_models', global_step=None):
        print("Saving...")
        # tl.files.save_npz(self.all_params, name=savedir + '/model.npz', sess=self.sess)
        self.saver.save(self.sess, savedir + "/model", global_step=global_step)
        print("Saved!")

    def resume(self, savedir='saved_models', global_step=None):

        if os.path.exists(savedir):
            if global_step is None:
                checkpoint_path_to_resume = tf.train.latest_checkpoint(savedir)
            else:
                checkpoint_path_to_resume = savedir + '/model-' + str(global_step)
                # checkpoint_path_to_resume = None
                # checkpoint_path_list = tf.train.get_checkpoint_state(savedir)
                # prefix_to_delete = checkpoint_path_list.model_checkpoint_path + '-'
                # global_step_str = str(global_step)
                # for checkpoint_path in checkpoint_path_list.all_model_checkpoint_paths:
                #     checkpoint_path_iteration = checkpoint_path.replace(prefix_to_delete,'')
                #     if(checkpoint_path_iteration == global_step_str):
                #         checkpoint_path_to_resume = checkpoint_path
                #         break
                #
                # if checkpoint_path_to_resume is None:
                #     checkpoint_path_to_resume = tf.train.latest_checkpoint(savedir)

            print("Restoring from " + checkpoint_path_to_resume)
            self.saver.restore(self.sess, checkpoint_path_to_resume)
            print("Restored!")

    def set_data(self, data):
        self.data = data

    def cycle_consistency_loss(self, real_images, generated_images):
        """Compute the cycle consistency loss.

        The cycle consistency loss is defined as the sum of the L1 distances
        between the real images from each domain and their generated (fake)
        counterparts.

        This definition is derived from Equation 2 in:
            Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
            Networks.
            Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros.


        Args:
            real_images: A batch of images from domain X, a `Tensor` of shape
                [batch_size, height, width, channels].
            generated_images: A batch of generated images made to look like they
                came from domain X, a `Tensor` of shape
                [batch_size, height, width, channels].

        Returns:
            The cycle consistency loss.
        """
        return tf.reduce_mean(tf.abs(real_images - generated_images))

    def lsgan_loss_generator(self, prob_fake_is_real):
        """Computes the LS-GAN loss as minimized by the generator.

        Rather than compute the negative loglikelihood, a least-squares loss is
        used to optimize the discriminators as per Equation 2 in:
            Least Squares Generative Adversarial Networks
            Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, and
            Stephen Paul Smolley.
            https://arxiv.org/pdf/1611.04076.pdf

        Args:
            prob_fake_is_real: The discriminator's estimate that generated images
                made to look like real images are real.

        Returns:
            The total LS-GAN loss.
        """
        return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))

    def lsgan_loss_discriminator(self, prob_real_is_real, prob_fake_is_real):
        """Computes the LS-GAN loss as minimized by the discriminator.

        Rather than compute the negative loglikelihood, a least-squares loss is
        used to optimize the discriminators as per Equation 2 in:
            Least Squares Generative Adversarial Networks
            Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, and
            Stephen Paul Smolley.
            https://arxiv.org/pdf/1611.04076.pdf

        Args:
            prob_real_is_real: The discriminator's estimate that images actually
                drawn from the real domain are in fact real.
            prob_fake_is_real: The discriminator's estimate that generated images
                made to look like real images are real.

        Returns:
            The total LS-GAN loss.
        """
        return (tf.reduce_mean(tf.squared_difference(prob_real_is_real, 1)) +
                tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 0))) * 0.5

    def latent_consistency_loss(self, real_images_z, generated_images_z):
        return tf.reduce_mean(tf.abs(real_images_z - generated_images_z))