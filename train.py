from xmudata.DIV2K2018 import DIV2K2018
from xmudata.sampledata import SampleData
from xmudata.adddata import AddData
import argparse
from xmumodel.edsr_deconv import EDSR
from xmumodel.cyclesr import CycleSR
import tensorflow as tf
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

FLAGS=None

def main(_):
    data = DIV2K2018(FLAGS.groundtruthdir, FLAGS.datadir, FLAGS.valid_groundtruthdir, FLAGS.valid_datadir,
                     FLAGS.imgsize, FLAGS.scale, FLAGS.postfixlen)
    # adddata = AddData(data, ratio=0.1)
    # sampledata = SampleData(data, sample_file=FLAGS.psnrpath, repeat_times=5, psnr_threshold=19)
    # network = EDSR(FLAGS.layers, FLAGS.featuresize, FLAGS.scale,channels=FLAGS.channels)
    network = CycleSR(FLAGS.featuresize, FLAGS.layers, FLAGS.channels)
    network.buildModel()
    network.set_data(data)
    network.train(FLAGS.batchsize, FLAGS.iterations, FLAGS.lr_init, FLAGS.lr_decay, FLAGS.decay_every,
                  FLAGS.savedir, False, FLAGS.reusedir, None, log_dir=FLAGS.logdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--groundtruthdir",default="data/DIV2K_2018/DIV2K_train_HR")
    """
    datadir                               postfix_len scale  track
    data/DIV2K_2018/DIV2K_train_LR_x8          2        8    1: bicubic downscaling x8 competition
    data/DIV2K_2018/DIV2K_train_LR_mild        3        4    2: realistic downscaling x4 with mild conditions competition
    data/DIV2K_2018/DIV2K_train_LR_difficult   3        4    3: realistic downscaling x4 with difficult conditions competition
    data/DIV2K_2018/DIV2K_train_LR_wild        4        4    4: wild downscaling x4 competition
    """
    parser.add_argument("--datadir",default="data/DIV2K_2018/DIV2K_train_LR_mild")
    parser.add_argument("--valid_groundtruthdir", default='data/DIV2K_2017/DIV2K_valid_HR')
    parser.add_argument("--valid_datadir", default="data/DIV2K_2018/DIV2K_valid_LR_mild")
    parser.add_argument("--postfixlen",default=3)
    parser.add_argument("--imgsize",default=48,type=int)
    parser.add_argument("--scale",default=4,type=int)
    parser.add_argument("--layers",default=16,type=int)
    parser.add_argument("--featuresize",default=128,type=int)
    parser.add_argument("--channels", default=3, type=int)
    parser.add_argument("--batchsize",default=10,type=int)
    parser.add_argument("--savedir",default='result/track2/cyclesr/ckpt')
    parser.add_argument("--logdir", default='result/track2/cyclesr/log')
    parser.add_argument("--reusedir",default='result/track2/22_898_v12500_cyclesr_v3/ckpt')
    parser.add_argument("--psnrpath", default='result/track2/22_898_v12500_cyclesr_v3/out/psnr_train.csv')
    parser.add_argument("--iterations",default=100000,type=int)
    parser.add_argument("--lr_init", default=1e-4)
    parser.add_argument("--lr_decay", default=0.5)
    parser.add_argument("--decay_every", default=200000)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
