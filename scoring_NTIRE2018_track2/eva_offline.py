#!/usr/bin/env python
import sys
import os
sys.path.append(os.getcwd())
import os.path
import random
import numpy as np

from PIL import Image
import scipy.misc
#from skimage.measure import structural_similarity as ssim
# from myssim import compare_ssim as ssim

SCALE = 4
SHIFT = 10
SIZE = 50


input_dir = '/usr/bendernas02/bendercache-a/sr/'


def output_measures(img_orig, img_out):
    # print type(img_out)
    h, w, c = img_orig.shape
    h_cen, w_cen = int(h / 2), int(w / 2)
    h_left = h_cen - SIZE
    h_right = h_cen + SIZE
    w_left = w_cen - SIZE
    w_right = w_cen + SIZE

    im_h = np.zeros([1, SIZE * 2, SIZE * 2, c])
    im_h[0, :, :, :] = img_orig[h_left:h_right, w_left:w_right, :]
    # ssim_h = np.squeeze(im_h)
    # print type(ssim_h)
    im_shifts = np.zeros([(2 * SHIFT + 1) * (2 * SHIFT + 1), SIZE * 2, SIZE * 2, c])
    ssim_shifts = np.zeros([(2 * SHIFT + 1) * (2 * SHIFT + 1), c])
    for hei in range(-SHIFT, SHIFT + 1):
        for wid in range(-SHIFT, SHIFT + 1):
            tmp_l = img_out[h_left + hei:h_right + hei, w_left + wid:w_right + wid, :]
            im_shifts[(hei + SHIFT) * (SHIFT + 1) + wid + SHIFT, :, :, :] = tmp_l

            # #ssim_h = np.squeeze(im_h)
            # ssim_h = ssim_h.astype('uint8')
            # ssim_l = tmp_l.astype('uint8')
            # if abs(hei) % 2 == 0 and abs(wid) % 2 == 0:
            #     for i in range(c):
            #         ssim_shifts[(hei + SHIFT) * (SHIFT + 1) + wid + SHIFT, i] \
            #             = ssim(ssim_l[:, :, i], ssim_h[:, :, i], gaussian_weights=True, use_sample_covariance=False)

    squared_error = np.square(im_shifts / 255. - im_h / 255.)
    mse = np.mean(squared_error, axis=(1, 2, 3))
    psnr = 10 * np.log10(1.0 / mse)
    return max(psnr), max(np.mean(ssim_shifts, axis=1))


def _open_img_measures(img_p):
    F = scipy.misc.fromimage(Image.open(img_p))#.astype(float)
    h, w, c = F.shape
    F = F[:h-h%SCALE, :w-w%SCALE, :]
    boundarypixels = 6+SCALE 
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
    return F

def compute_measures(ref_im, res_im):
    return output_measures(
        _open_img_measures(os.path.join(input_dir,'DIV2K_valid_HR',ref_im)),
        _open_img_measures(os.path.join(input_dir,'DIV2K_valid_fake_wild',res_im))
        )


# as per the metadata file, input and output directories are the arguments


ref_dir = os.path.join(input_dir, 'DIV2K_valid_HR/')
res_dir = os.path.join(input_dir, 'DIV2K_valid_fake_wild/')


ref_pngs = sorted([p for p in os.listdir(ref_dir) if p.lower().endswith('png')])
res_pngs = sorted([p for p in os.listdir(res_dir) if p.lower().endswith('png')])
if not (len(ref_pngs)==100 and len(res_pngs)==100):
    raise Exception('Expected 100 .png images, got %d'%len(res_pngs))


scores_psnr = []
scores_ssim = []
for (ref_im, res_im) in zip(ref_pngs, res_pngs):
    print(ref_im, res_im)
    best_psnr, best_ssim = compute_measures(ref_im, res_im)
    # print best_psnr
    # print best_ssim
    scores_psnr.append(best_psnr)
    scores_ssim.append(best_ssim)

psnr = np.mean(scores_psnr)
mssim = np.mean(scores_ssim)



print(psnr)
print(mssim)