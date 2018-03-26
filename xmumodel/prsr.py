from xmumodel.model import Model
import tensorflow as tf
import tensorlayer.layers as tl
from xmuutil.mask_conv_layer import MaskConvLayer
from xmuutil.gated_cnn_layer import GatedCNNLayer
from xmuutil.relulayer import ReluLayer
from xmuutil.scalelayer import ScaleLayer
from xmuutil.transposed_conv2d_layer import TransposedConv2dLayer
from xmuutil import utils
from tqdm import tqdm
import os
import shutil


class PixelResNet(Model):
    """
    Pixel Recursive Super Resolution Network implementation
    From https://arxiv.org/pdf/1702.00783.pdf
    """
    def build_model(self):
        lr_images = self._normalize_color(self.input)
        hr_images = self._normalize_color(self.target)

        conditioning_logits = self.conditioning_network(lr_images)
        prior_logits = self.prior_network(hr_images)

        self.output = conditioning_logits
        self.calculate_loss(conditioning_logits + prior_logits)

        # Tensorflow graph setup... session, saver, etc.
        conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=conf)
        self.saver = tf.train.Saver()
        print("Done building!")

    def prior_network(self, hr_images=None):
        """
        Create PixelCNN prior network
        input: hr_images with shape [batch_size, hr_height, hr_width, in_channel]
        :return:
        prior_logits: [batch_size, hr_height, hr_width, 3*256]
        """
        x = tl.InputLayer(hr_images, name='prior_input')
        x = MaskConvLayer(x, 64, [7, 7], mask_type='a', name="prior_conv1")
        for i in range(20):
            x = GatedCNNLayer(x, [5, 5], name='gated'+str(i))
        x = MaskConvLayer(x, 1024, [1,1], mask_type='b', name='prior_conv2')
        x = ReluLayer(x, name='relu1')
        prior_logits = MaskConvLayer(x, 3 * 256, [1, 1], mask_type='b', name="prior_conv3").outputs
        prior_logits = tf.concat([prior_logits[:, :, :, 0::3],
                                  prior_logits[:, :, :, 1::3],
                                  prior_logits[:, :, :, 2::3]], axis=-1)
        return prior_logits

    def conditioning_network(self, lr_images):
        """
        Create ResNet conditioning network
        input: lr_images with shape [batch_size, lr_height, lr_width, in_channels]
        :return:
        conditioning_logits: [batch_size, hr_height, hr_width, 3*256]
        """
        x = tl.InputLayer(lr_images, name='condition_input')
        x = tl.Conv2d(x, 32, [1, 1], name='condition_c1')
        for i in range(2):
            for j in range(self.num_layers):
                x = self.__resBlock(x, 32, (3, 3), num_layer=(i*self.num_layers+j))
            x = TransposedConv2dLayer(x, 32, [3, 3],strides=[2,2], name='condition_deconv_' + str(i))
            x = ReluLayer(x, name='condition_relu_' + str(i))

        for i in range(self.num_layers):
            x = self.__resBlock(x, 32, (3, 3), num_layer=(2*self.num_layers+i))

        conditioning_logits = tl.Conv2d(x, 3 * 256, (1,1), name='condition_c2').outputs
        return conditioning_logits

    def __resBlock(self, x, channels=64, kernel_size=(3,3), scale=1, num_layer=0):
        nn = tl.Conv2d(x, channels, kernel_size, act=None, name='res%d/c1' % (num_layer))
        nn = tl.BatchNormLayer(nn,act=tf.nn.relu, name='res%d/b1' % (num_layer))
        nn = tl.Conv2d(nn, channels, kernel_size, act=None, name='res%d/c2' % (num_layer))
        nn = tl.BatchNormLayer(nn, name='res%d/b2' % (num_layer))
        nn = ScaleLayer(nn, scale, name='res%d/scale' % (num_layer))
        n = tl.ElementwiseLayer([x, nn], tf.add, name='res%d/res_add' % (num_layer))
        return n

    def _normalize_color(self, image):
        """
        Rescale pixel color intensity to [-0.5, 0.5]
        """
        return image / 255.0 - 0.5

    def calculate_loss(self, output):
        self.loss = utils.softmax_cross_entropy_loss(output, self.target)

        # Scalar to keep track for loss
        summary_loss = tf.summary.scalar("loss", self.loss)

        streaming_loss, self.streaming_loss_update = tf.contrib.metrics.streaming_mean(self.loss)
        streaming_loss_scalar = tf.summary.scalar('loss',streaming_loss)

        # # Image summaries for input, target, and output
        # input_image = tf.summary.image("input_image", tf.cast(self.input, tf.uint8))
        # target_image = tf.summary.image("target_image", tf.cast(self.target, tf.uint8))
        # output_image = tf.summary.image("output_image", tf.cast(x.outputs, tf.uint8))
        #
        self.train_merge = tf.summary.merge([summary_loss])
        self.test_merge = tf.summary.merge([streaming_loss_scalar])

    def train(self,batch_size= 10, iterations=1000, lr_init=1e-4, lr_decay=0.5, decay_every=2e5,
              save_dir="saved_models",reuse=False,reuse_dir=None,log_dir="log"):
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
        # Using adam optimizer as mentioned in the paper
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_v)
        # optimizer = tf.train.RMSPropOptimizer(lr_v, decay=0.95, momentum=0.9, epsilon=1e-8)
        # This is the train operation for our objective
        self.train_op = optimizer.minimize(self.loss)


        #Operation to initialize all variables
        init = tf.global_variables_initializer()
        print("Begin training...")
        with self.sess as sess:
            #Initialize all variables
            sess.run(init)
            if reuse:
                self.resume(reuse_dir)

            #create summary writer for train
            train_writer = tf.summary.FileWriter(log_dir+"/train",sess.graph)

            #If we're using a test set, include another summary writer for that
            test_writer = tf.summary.FileWriter(log_dir+"/test",sess.graph)
            test_feed = []
            while True:
                test_x,test_y = self.data.get_test_set(batch_size)
                if test_x!=None and test_y!=None:
                    test_feed.append({
                            self.input:test_x,
                            self.target:test_y
                    })
                else:
                    break

            sess.run(tf.assign(lr_v, lr_init))
            #This is our training loop
            for i in tqdm(range(iterations)):
                if i != 0 and (i % decay_every == 0):
                    new_lr_decay = lr_decay ** (i // decay_every)
                    sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                #Use the data function we were passed to get a batch every iteration
                x,y = self.data.get_batch(batch_size, i)
                #Create feed dictionary for the batch
                feed = {
                    self.input:x,
                    self.target:y
                }
                #Run the train op and calculate the train summary
                summary,_ = sess.run([self.train_merge,self.train_op],feed)
                #Write train summary for this step
                train_writer.add_summary(summary,i)
                #test every 10 iterations
                if i%100 == 0:
                    sess.run(tf.local_variables_initializer())
                    for j in range(len(test_feed)):
                        # sess.run([self.streaming_loss_update],feed_dict=test_feed[j])
                        sess.run([self.streaming_loss_update], feed_dict=test_feed[j])
                    streaming_summ = sess.run(self.test_merge)
                    #Write test summary
                    test_writer.add_summary(streaming_summ,i)

                # Save our trained model
                if (i!=0 and i % 500 == 0) or (i+1 == iterations):
                    self.save(save_dir)

            test_writer.close()
            train_writer.close()
