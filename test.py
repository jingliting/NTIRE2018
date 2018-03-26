import argparse
from xmumodel.edsr import EDSR
import tensorflow as tf
import tensorlayer as tl
import os
import sys
from xmuutil import utils
import numpy as np
from collections import OrderedDict

FLAGS=None

def main(_):
    img_files = sorted(os.listdir(FLAGS.datadir))
    lr_imgs = tl.visualize.read_images(img_files, FLAGS.datadir)

    mean_list = []
    dict = {}
    for i in range(len(img_files)):
        name_lr, postfix = os.path.splitext(img_files[i])
        name_hr = name_lr
        hr_img = tl.visualize.read_image(name_hr + postfix, FLAGS.grouthtruth)
        mean = utils.psnr_np(hr_img,lr_imgs[i])
        print('%d -> %s : %.6f'%(i,name_lr + postfix,mean))
        mean_list.append(mean)
        dict[name_hr] = mean

    dict_sort_by_value = OrderedDict(sorted(dict.items(),key=lambda x:x[1]))
    mean = np.mean(mean_list)
    with open(FLAGS.datadir + '/' + FLAGS.record,'w') as file:
        for k,v in dict_sort_by_value.items():
            file.write('%s : %.6f\n'%(k,v))
        file.write('Average: %.6f\n'%(mean))

    print('Average: %.6f'%(mean))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", default='out/track2_edsr_deconv')
    parser.add_argument("--grouthtruth",default='data/DIV2K_2017/DIV2K_valid_HR')
    parser.add_argument("--postfixlen", default=2,type=int)
    parser.add_argument("--record", default='recordfile.txt')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
