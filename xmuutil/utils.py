import tensorflow as tf
import numpy as np
from xmuutil.exception import LargeSizeException
import tensorlayer.layers as tl
from scipy.ndimage import rotate
import pywt
# import cv2

def log10(x):
    """
    Tensorflow log base 10.
    Found here: https://github.com/tensorflow/tensorflow/issues/1666
    """
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def crop(img_lr, img_hr, wrg, hrg, scale = 8,is_random=True, row_index=0, col_index=1, channel_index=2):
    """Randomly or centrally crop an image.

    Parameters
    ----------
    img_lr,img_hr : numpy array
        An image with dimension of [row, col, channel] (default).
    wrg : int
        Size of low resolution image width.
    hrg : int
        Size of low resolution image  height.
    scale : int
        upsample scale
    is_random : boolean, default False
        If True, randomly crop, else central crop.
    row_index, col_index, channel_index : int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    """
    h, w = img_lr.shape[row_index], img_lr.shape[col_index]
    h_hr, w_hr = img_hr.shape[row_index], img_hr.shape[col_index]
    if (hrg > h) or (wrg > w):
        raise LargeSizeException("The size of cropping file should smaller than the original image")

    if (hrg*scale > h_hr ) or (wrg*scale > w_hr):
        raise LargeSizeException("The scaled size of cropping should smaller than the ground truth image")

    if is_random:
        h_offset = int(np.random.uniform(0, h - hrg) - 1)
        w_offset = int(np.random.uniform(0, w - wrg) - 1)
        # print(h_offset, w_offset, x[h_offset: hrg+h_offset ,w_offset: wrg+w_offset].shape)
        return img_lr[h_offset: hrg + h_offset, w_offset: wrg + w_offset], \
               img_hr[h_offset*scale:(hrg + h_offset)*scale, w_offset*scale: (wrg + w_offset)*scale]
    else:  # central crop
        h_offset = int(np.floor((h - hrg) / 2.))
        w_offset = int(np.floor((w - wrg) / 2.))
        h_end = h_offset + hrg
        w_end = w_offset + wrg
        return img_lr[h_offset: h_end, w_offset: w_end], \
               img_hr[h_offset*scale:h_end*scale, w_offset*scale: w_end*scale]

    tf.image.crop_and_resize()


def subpixelupsample(x, feature_size, scale = 4):
    # Upsample output of the convolution
    x = tl.Conv2d(x, feature_size, [3, 3], act = None, name = 's1/1')
    x = tl.SubpixelConv2d(x, scale = 2, act=tf.nn.relu, name='pixelshufferx2/1')

    x = tl.Conv2d(x, feature_size, [3, 3], act = None, name = 's1/2')
    x = tl.SubpixelConv2d(x, scale = 2, act=tf.nn.relu, name='pixelshufferx2/2')

    if scale > 4:
        #X8
        x = tl.Conv2d(x, feature_size, [3, 3], act = None, name = 's1/3')
        x = tl.SubpixelConv2d(x, scale = 2, act=tf.nn.relu, name='pixelshufferx2/3')
    return x


def split_to_separate_channel(data_x):
    if data_x == None:
        return None
    separate_data_x = []
    for i in range(len(data_x)):
        split_data_x = np.split(data_x[i],indices_or_sections=3,axis=2)
        for j in range(3):
            separate_data_x.append(split_data_x[j])
    return separate_data_x


def concat_separate_channel(data):
    return np.transpose(data,[3,1,2,0])

# def get_dwt_images(img_list):
#     """
#     use wavelet to decomposite the single-channel images
#     :param img_list the single-channel image
#     :return the images with four channels, which are cA, cH, cV and cD
#                                -------------------
#                                |        |        |
#                                | cA(LL) | cH(LH) |
#                                |        |        |
#    (cA, (cH, cV, cD))  <--->   -------------------
#                                |        |        |
#                                | cV(HL) | cD(HH) |
#                                |        |        |
#                                -------------------
#
#            (DWT 2D output and interpretation)
#    """
#     dwt_imgs = []
#     # output the imgs
#     for i in range(len(img_list)):
#         img = img_list[i][:, :, 0]
#         cA,(cH,cV,cD) = pywt.dwt2(img,'db1')
#         wt_img = cv2.merge((cA,cH,cV,cD))
#         dwt_imgs.append(wt_img)
#     return dwt_imgs

def dwt_compose(dwtimg_list):
    """
    compose the single-channel image from the cA,cH,cV,cD
    :param dwtimg_list: 4-channel imgs (cA,cH,cV,cD)
    :return: composed images
    """
    imgs = []
    for i in range(len(dwtimg_list)):
        cA = dwtimg_list[i][:,:,0]
        cH = dwtimg_list[i][:,:,1]
        cV = dwtimg_list[i][:, :, 2]
        cD = dwtimg_list[i][:, :, 3]
        coef = [cA,(cH, cV, cD)]
        img = pywt.idwt2(coef, 'db1')
        imgs.append(img)
    return imgs

def psnr_tf(target, output, target_height=256, target_width=256, scale=None, is_norm=False):
    """
    Calculating Peak Signal-to-noise-ratio
    Using equations from here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    target: 4-D Tensor of shape `[batch, height, width, channels]
    output: 4-D Tensor of shape `[batch, height, width, channels]
    target_height: Height of target/output image
    target_width: Width of target/output image
    scale: 6+scale pixels to ignore
    """
    if scale:
        boundarypixels = scale + 6
        target = tf.image.crop_to_bounding_box(target, boundarypixels + 1, boundarypixels + 1,
                                                target_height - 2 * boundarypixels, target_width - 2 * boundarypixels)
        output = tf.image.crop_to_bounding_box(output, boundarypixels + 1, boundarypixels + 1,
                                                target_height - 2 * boundarypixels, target_width - 2 * boundarypixels)
    if not is_norm:
        target = tf.divide(target,tf.constant(255.0, dtype=tf.float32))
        output = tf.divide(output,tf.constant(255.0, dtype=tf.float32))

    mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(target, output), [1, 2, 3]))
    PSNR = tf.constant(1.0, dtype=tf.float32) / mse
    PSNR = tf.constant(10, dtype=tf.float32) * log10(PSNR)

    return PSNR


def enhance_imgs(imgs_list, rotate_rg=0, flip_axis=0, is_swirl=False):
    """
    :param imgs_list: List of images with dimension of [n_images, row, col, channel] (default).
    :param rotate_rg: int. Degree to rotate, 0, 90
    :param flip_axis: int
        - 0, flip up and down
        - 1, flip left and right
        - 2, flip up and down first, then flip left and right
        - -1, no flip
    :param is_swirl: True/False
    :return:
    """
    if imgs_list is None:
        return None
    imgs = imgs_list
    imgs = [rotate(img, rotate_rg) for img in imgs]
    if flip_axis == 0 or flip_axis == 1:
        imgs = tl.visualize.prepro.flip_axis_multi(imgs, axis=flip_axis, is_random=False)
    elif flip_axis == 2:
        imgs = tl.visualize.prepro.flip_axis_multi(imgs, axis=0, is_random=False)
        imgs = tl.visualize.prepro.flip_axis_multi(imgs, axis=1, is_random=False)
    if is_swirl:
        imgs = tl.visualize.prepro.swirl_multi(imgs, is_random=True)
    return imgs


def anti_enhance_imgs(imgs_list, rotate_rg=0, flip_axis=0):
    """
    Reverse operation of rotating and flipping images
    """
    if imgs_list is None:
        return None
    imgs = imgs_list
    if flip_axis == 0 or flip_axis == 1:
        imgs = tl.visualize.prepro.flip_axis_multi(imgs, axis=flip_axis, is_random=False)
    elif flip_axis == 2:
        imgs = tl.visualize.prepro.flip_axis_multi(imgs, axis=0, is_random=False)
        imgs = tl.visualize.prepro.flip_axis_multi(imgs, axis=1, is_random=False)
    imgs = [rotate(img, 360-rotate_rg) for img in imgs]
    return imgs


def psnr_np(target, output, scale=None):
    """
    Numpy implementation of PSNR for single image
    """
    if np.max(target) > 1.0:
        target = normalize_color([target])[0]
    if np.max(output) > 1.0:
        output = normalize_color([output])[0]
    output_h, output_w, _ = output.shape
    target_h, target_w, _ = target.shape
    if target_h!=output_h or target_w!=output_w:
        min_h = np.minimum(target_h, output_h)
        min_w = np.minimum(target_w, output_w)
        target = target[:min_h, :min_w, :]
        output = output[:min_h, :min_w, :]

    if scale:
        boundarypixels = scale + 6
        h, w, _ = target.shape
        target = target[boundarypixels + 1:h - boundarypixels, boundarypixels + 1:w - boundarypixels, :]
        output = output[boundarypixels + 1:h - boundarypixels, boundarypixels + 1:w - boundarypixels, :]
    mse = np.mean(np.square(target - output))
    PSNR = 1.0 / mse
    PSNR = 10 * np.log10(PSNR)
    return PSNR


def psnr_np_wild(target, output, shift=10, size = 50):
    """
    Numpy implementation of PSNR for single image
    :param target,the target of HR image which is 3-dimensions
    :param output,the output of model which is 3-dimensions
    """
    if np.max(target) > 1.0:
        target = normalize_color([target])[0]
    if np.max(output) > 1.0:
        output = normalize_color([output])[0]

    target_h, target_w, target_c = target.shape
    h_center = target_h // 2
    w_center = target_w // 2

    h_left = h_center - size
    h_right = h_center + size
    w_left = w_center - size
    w_right = w_center + size

    target_center = target[h_left:h_right,w_left:w_right,:]
    target_center = np.reshape(target_center,[1,-1])

    output_shift = np.zeros(shape=[(2*shift+1)*(2*shift+1),size * size * 4 * target_c])
    for i in range(-shift,shift + 1 , 1):
        for j in range(-shift,shift + 1, 1):
            output_temp = output[h_left + i:h_right + i,w_left + j:w_right+j,:]
            output_shift[(i + shift) * (2 * shift + 1) + j + shift:] = np.reshape(output_temp,[1,-1])[0]

    error = target_center - output_shift
    psnr_all = 10 * np.log10(1.0 / np.mean(np.square(error),1))
    return np.max(psnr_all)



def learning_rate_decay(learning_rate, global_step, decay_rate = 5e-5, name=None,sess=None):
    '''Adapted from https://github.com/torch/optim/blob/master/adam.lua'''
    if global_step is None:
        raise ValueError("global_step is required for exponential_decay.")
    print(global_step)
    with tf.name_scope(name, "ExponentialDecay", [learning_rate, global_step, decay_rate]) as name:
        learning_rate = tf.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = tf.cast(global_step, dtype)
        decay_rate = tf.cast(decay_rate, dtype)

        # local clr = lr / (1 + state.t*lrd)
        return learning_rate / (1 + global_step*decay_rate)


def normalize_color(imgs_list):
    """
    Helper to rescale pixel color intensity to [0, 1]
    """
    if imgs_list is None:
        return None
    norm_imgs_list = [img / 255.0 for img in imgs_list]
    return norm_imgs_list

def normalize_color_tf(imgs):
    """
    Helper to rescale pixel color intensity to [0, 1]
    imgs: tensor of bhwc
    """
    if imgs is None:
        return None
    return tf.divide(imgs, tf.constant(255.0, tf.float32))


def softmax_cross_entropy_loss(logits, labels):
    """
    Compute cross_entropy loss
    :param logits: sum from conditioning and prior networks
    :param labels: ground truth images
    :return: cross_entropy loss over image
    """
    logits = tf.reshape(logits, [-1, 256])
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [-1])
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return loss