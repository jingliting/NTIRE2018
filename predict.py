import argparse
from xmumodel.edsr import EDSR
from xmudata.preddata import data_for_predict
import tensorflow as tf
import tensorlayer as tl
import sys
from xmuutil import utils
from tqdm import tqdm
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

FLAGS = None


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

    network = EDSR(FLAGS.n_channels)
    network.build_model(FLAGS.n_res_blocks, FLAGS.n_features, FLAGS.scale)
    network.resume(FLAGS.reusedir, None)

    hr_list, lr_imgs, groundtruth_imgs = data_for_predict(FLAGS.datadir, FLAGS.grouthtruth, FLAGS.postfixlen)

    # Valid
    if groundtruth_imgs:
        psnr_list = []
        time_list = []
        fo = open(FLAGS.outdir + '/psnr.csv', 'w')
        fo.writelines("file, PSNR\n")
        for lr_img, groundtruth_img, hr_name in zip(lr_imgs, groundtruth_imgs, hr_list):
            start = time.time()
            out = network.predict([lr_img])
            # out = enhance_predict([lr_img],network)
            use_time = time.time() - start
            time_list.append(use_time)
            tl.vis.save_image(out[0], FLAGS.outdir + '/' + hr_name)
            psnr = utils.psnr_np(groundtruth_img, out[0], scale=4)
            print('%s : %.6f' % (hr_name, psnr))
            psnr_list.append(psnr)
            fo.writelines("%s, %.6f\n" % (hr_name, psnr))

        print(np.mean(psnr_list))
        print(np.mean(time_list))
        fo.writelines("Average psnr,0, %.6f" % (np.mean(psnr_list)))
        fo.writelines("Average runtime,0, %.6f" % (np.mean(time_list)))
        fo.close()

    # Test
    else:
        for i in tqdm(range((len(hr_list)))):
            # out = network.predict([lr_imgs[i]])
            out = enhance_predict([lr_imgs[i]], network)
            tl.vis.save_image(out[0], FLAGS.outdir + '/' + hr_list[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", default='data/minic_LR_x8/')
    parser.add_argument("--groundtruth", default='data/minic_HR')
    parser.add_argument("--postfixlen", default=2, type=int)
    parser.add_argument("--scale", default=8, type=int)
    parser.add_argument("--n_res_blocks", default=32, type=int)
    parser.add_argument("--n_features", default=128, type=int)
    parser.add_argument("--reusedir", default='ckpt/edsr_z16_test_lr')
    parser.add_argument("--outdir", default='out/edsr_z16_test_lr')
    parser.add_argument("--samplefile", default='sample.txt')
    parser.add_argument("--n_channels", default=3, type=int)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
