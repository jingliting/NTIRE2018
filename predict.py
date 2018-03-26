import argparse
from xmumodel.edsr import EDSR
from xmudata.preddata import data_for_predict
import tensorflow as tf
import tensorlayer as tl
import sys
from xmuutil import utils
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

FLAGS=None


def enhance_predict(lr_imgs, network=None):
    outs_list = []
    for _, flip_axis in enumerate([0, 1, 2, -1]):
        for _, rotate_rg in enumerate([0, 90]):
            en_imgs = utils.enhance_imgs(lr_imgs,  rotate_rg, flip_axis)
            outs = network.predict(en_imgs)
            anti_outs = utils.anti_enhance_imgs(outs, rotate_rg, flip_axis)
            outs_list.append(anti_outs)
    return np.mean(outs_list, axis=0)


def main(_):
    if not os.path.exists(FLAGS.outdir):
        os.mkdir(FLAGS.outdir)

    network = EDSR(FLAGS.layers, FLAGS.featuresize, FLAGS.scale, FLAGS.channels)
    network.buildModel()
    network.resume(FLAGS.reusedir)

    hr_list, lr_imgs, groundtruth_imgs = data_for_predict(FLAGS.datadir, FLAGS.grouthtruth, FLAGS.postfixlen)

    if groundtruth_imgs:
        psnr_list = []
        sample_list = []
        for lr_img, groundtruth_img, hr_name in zip(lr_imgs, groundtruth_imgs, hr_list):
            out = network.predict([lr_img / 255.0])
            # tl.vis.save_image(hr_img[0], FLAGS.outdir + '/' + hr_name)
            psnr = utils.psnr_np(groundtruth_img, out[0]*255, scale=6, is_norm=False)
            # print('%s : %.6f' % (hr_name, psnr))
            psnr_list.append(psnr)
            if psnr < 20.0:
                sample_list.append(hr_name)

        print(np.mean(psnr_list))
        with open(FLAGS.outdir+"/"+FLAGS.sample_file, 'w') as f:
            f.write('\n'.join(sample_list))

    else:
        for i in range(len(hr_list)):
            out = network.predict([lr_imgs[i]])
            tl.vis.save_image(out[0], FLAGS.outdir + '/' + hr_list[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", default='data/minic_LR_x8/')
    parser.add_argument("--grouthtruth",default='data/minic_HR')
    parser.add_argument("--postfixlen", default=2,type=int)
    parser.add_argument("--scale",default=8,type=int)
    parser.add_argument("--layers",default=32,type=int)
    parser.add_argument("--featuresize",default=128,type=int)
    parser.add_argument("--reusedir",default='ckpt/edsr_z16_test_lr')
    parser.add_argument("--outdir", default='out/edsr_z16_test_lr')
    parser.add_argument("--samplefile", default='sample.txt')
    parser.add_argument("--channels",default=3,type=int)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
