import tensorflow as tf
import argparse
import sys

FLAGS = None


# Export trained model and save as .pb file
def main(_):
    with tf.Session() as sess:

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Get the latest checkpoint
        latest_ckpt = tf.train.latest_checkpoint(FLAGS.reuse_dir)

        # Import graph
        restore_saver = tf.train.import_meta_graph(FLAGS.graph_path)

        # Restore graph
        restore_saver.restore(sess, latest_ckpt)

        # Convert_variables_to_constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, [FLAGS.output])

        # Save new graph
        with tf.gfile.FastGFile(FLAGS.save_path, mode='wb') as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--reuse_dir", default="result/track1/ckpt")
    parser.add_argument("--graph_path", default="result/track1/ckpt/model-0.meta")
    parser.add_argument("--output", default="output")
    parser.add_argument("--save_path", default="result/track1/edsr_x8.pb")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
