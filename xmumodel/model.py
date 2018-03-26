import tensorflow as tf
from tqdm import tqdm
from abc import ABCMeta,abstractmethod
import os
import shutil
from xmuutil import utils


"""
An implementation of the neural network used for
super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""
class Model(object, metaclass=ABCMeta):
    def __init__(self, num_layers=32, feature_size=256, scale=8, channels=3):
        self.num_layers = num_layers
        self.feature_size = feature_size
        self.scale = scale
        self.output_channels = channels

        #Placeholder for image inputs
        self.input = tf.placeholder(tf.float32, [None, None, None, channels])
        #Placeholder for upscaled image ground-truth
        self.target = tf.placeholder(tf.float32, [None, None, None, channels])
        self.output = None

        self.train_op = None

    @abstractmethod
    def buildModel(self):
        pass

    """
    Save the current state of the network to file
    """
    def save(self, savedir='saved_models', global_step=None):
        print("Saving...")
        # tl.files.save_npz(self.all_params, name=savedir + '/model.npz', sess=self.sess)
        self.saver.save(self.sess,savedir+"/model", global_step=global_step)
        print("Saved!")

    """
    Resume network from previously saved weights
    :param global_step, the specific iteration tot resume
    if global_step is None,resume the latest checkpoint
    First to judge whether the global_step exists in savedir,
    if not exists, resume the latest checkpoint default.
    Otherwise, resume the specific iteration
    """
    def resume(self,savedir='saved_models',global_step=None):

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
            self.saver.restore(self.sess,checkpoint_path_to_resume)
            print("Restored!")


    def cacuLoss(self, x):
        self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.target, x.outputs))

        PSNR = utils.psnr_tf(self.target, x.outputs, is_norm=True)

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

    """
    Function to setup your input data pipeline
    """
    def set_data(self, data):
        self.data = data


    """
    Estimate the trained model
    x: (tf.float32, [batch_size, h, w, output_channels])
    """
    def predict(self, x):
        return self.sess.run(self.output, feed_dict={self.input: x})

    """
    Train the neural network
    """
    def train(self,batch_size= 10, iterations=1000, lr_init=1e-4, lr_decay=0.5, decay_every=2e5,
              save_dir="saved_models",reuse=False,reuse_dir=None, reuse_step=1, log_dir="log"):
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
                self.resume(reuse_dir, reuse_step)

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
                #test every 10 iterations
                if i%100 == 0:
                    sess.run(tf.local_variables_initializer())
                    for j in range(len(test_feed)):
                        # sess.run([self.streaming_loss_update],feed_dict=test_feed[j])
                        sess.run([self.streaming_loss_update, self.streaming_psnr_update], feed_dict=test_feed[j])
                    streaming_summ = sess.run(self.test_merge)
                    #Write test summary
                    test_writer.add_summary(streaming_summ,i)

                # Save our trained model
                if (i!=0 and i % 500 == 0) or (i+1 == iterations):
                    self.save(save_dir, i)

            test_writer.close()
            train_writer.close()