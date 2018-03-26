
from xmumodel.model import Model
import tensorlayer.layers as tl
import tensorflow as tf
from xmuutil.scalelayer import ScaleLayer
from xmuutil.relulayer import ReluLayer
from xmuutil.transposed_conv2d_layer import TransposedConv2dLayer
from xmuutil.normalizelayer import NormalizeLayer
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
class EDSR(Model):
    def buildModel(self):
        print("Building EDSR...")

        self.norm_input = utils.normalize_color_tf(self.input)
        self.norm_target = utils.normalize_color_tf(self.target)

        #input layer
        x = tl.InputLayer(self.norm_input, name='inputlayer')
        # x = NormalizeLayer(x, name='normlayer')

        # One convolution before res blocks and to convert to required feature depth
        x = tl.Conv2d(x, self.feature_size, [3, 3], name='c')

        # Store the output of the first convolution to add later
        conv_1 = x

        scaling_factor = 0.1

        # Add the residual blocks to the model
        for i in range(self.num_layers):
            x = self.__resBlock(x, self.feature_size, scale=scaling_factor,layer=i)

        # One more convolution, and then we add the output of our first conv layer
        x = tl.Conv2d(x, self.feature_size, [3, 3], act = None, name = 'm1')
        x = tl.ElementwiseLayer([conv_1,x],tf.add, name='res_add')

        x = TransposedConv2dLayer(x, self.feature_size // 2, [5, 5], [2, 2], name='deconv_1')
        x = tl.Conv2d(x, self.feature_size, [3, 3], act=tf.nn.relu, name='deconv_conv_1')
        x = TransposedConv2dLayer(x, self.feature_size // 2, [5, 5], [2, 2], name='deconv_2')

        # One final convolution on the upsampling output
        output = tl.Conv2d(x,self.output_channels,[1,1],act=tf.nn.relu, name='lastLayer')
        self.output = output.outputs
        self.output = tf.clip_by_value(output.outputs, 0.0, 1.0)

        self.cacuLoss()

        # Tensorflow graph setup... session, saver, etc.
        conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=conf)
        self.saver = tf.train.Saver(max_to_keep=100)
        print("Done building!")


    def __resBlock(self, x, channels = 64, kernel_size = (3, 3), scale = 1.0,layer = 0):
        nn = ReluLayer(x, name='res%d/ru1'%(layer))
        nn = tl.Conv2d(nn, channels, kernel_size,name='res%d/c1'%(layer))
        nn = ReluLayer(nn, name='res%d/ru2' % (layer))
        nn = tl.Conv2d(nn, channels, kernel_size, name='res%d/c2' % (layer))
        nn = ScaleLayer(nn,scale, name='res%d/scale'%(layer))
        n = tl.ElementwiseLayer([x,nn],tf.add, name='res%d/res_add'%(layer))
        return n

    def cacuLoss(self):
        self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.norm_target, self.output))

        PSNR = utils.psnr_tf(self.norm_target, self.output, is_norm=True)

        # Scalar to keep track for loss
        summary_loss = tf.summary.scalar("loss", self.loss)
        summary_psnr = tf.summary.scalar("PSNR", PSNR)

        streaming_loss, self.streaming_loss_update = tf.contrib.metrics.streaming_mean(self.loss)
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
                self.resume(reuse_dir,global_step=reuse_step)

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
                x,y = self.data.get_batch(batch_size)
                #Create feed dictionary for the batch
                feed = {
                    self.input:x,
                    self.target:y
                }
                #Run the train op and calculate the train summary
                summary,_ = sess.run([self.train_merge,self.train_op],feed)
                #Write train summary for this step
                train_writer.add_summary(summary,i)

                # Test every 500 iterations and save our trained model
                if (i!=0 and i % 500 == 0) or (i+1 == iterations):
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