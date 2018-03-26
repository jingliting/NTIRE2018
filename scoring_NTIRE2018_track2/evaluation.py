#!/usr/bin/env python
import sys
import os
import os.path
import random
import numpy as np

from PIL import Image
import scipy.misc
#from skimage.measure import structural_similarity as ssim
# from myssim import compare_ssim as ssim


SCALE = 4
SHIFT = 40
SIZE = 30

def output_measures(img_orig, img_out):
    h, w, c = img_orig.shape
    h_cen, w_cen = int(h / 2), int(w / 2)
    h_left = h_cen - SIZE
    h_right = h_cen + SIZE
    w_left = w_cen - SIZE
    w_right = w_cen + SIZE

    im_h = np.zeros([1, SIZE * 2, SIZE * 2, c])
    im_h[0, :, :, :] = img_orig[h_left:h_right, w_left:w_right, :]
    ssim_h = np.squeeze(im_h)
    im_shifts = np.zeros([(2 * SHIFT + 1) * (2 * SHIFT + 1), SIZE * 2, SIZE * 2, c])
    ssim_shifts = np.zeros([(2 * SHIFT + 1) * (2 * SHIFT + 1), c])
    for hei in range(-SHIFT, SHIFT + 1):
        for wid in range(-SHIFT, SHIFT + 1):
            tmp_l = img_out[h_left + hei:h_right + hei, w_left + wid:w_right + wid, :]
            mean_l = np.mean(tmp_l)
            mean_o = np.mean(img_orig[h_left:h_right, w_left:w_right, :])
            im_shifts[(hei + SHIFT) * (2 * SHIFT + 1) + wid + SHIFT, :, :, :] = tmp_l/mean_l*mean_o

	    # #ssim_h = np.squeeze(im_h)
         #    ssim_h = ssim_h.astype('uint8')
         #    ssim_l = tmp_l.astype('uint8')
         #    if abs(hei) % 2 == 0 and abs(wid) % 2 == 0:
         #        for i in range(c):
         #            ssim_shifts[(hei + SHIFT) * (2 * SHIFT + 1) + wid + SHIFT, i] \
         #                = ssim(ssim_l[:, :, i], ssim_h[:, :, i], gaussian_weights=True, use_sample_covariance=False)

    squared_error = np.square(im_shifts / 255. - im_h / 255.)
    mse = np.mean(squared_error, axis=(1, 2, 3))
    psnr = 10 * np.log10(1.0 / mse)
    return max(psnr), max(np.mean(ssim_shifts, axis=1))


def _open_img_measures(img_p):
    F = scipy.misc.fromimage(Image.open(img_p))  # astype(float)

    h, w, c = F.shape
    F = F[:h-h%SCALE, :w-w%SCALE, :]
    boundarypixels = 6+SCALE
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
    return F
#
# def compute_measures(ref_im, res_im):
#     return output_measures(
#         _open_img_measures(os.path.join(input_dir,'ref',ref_im)),
#         _open_img_measures(os.path.join(input_dir,'res',res_im))
#         )
#
#
# # as per the metadata file, input and output directories are the arguments
# [_, input_dir, output_dir] = sys.argv
#
# res_dir = os.path.join(input_dir, 'res/')
# ref_dir = os.path.join(input_dir, 'ref/')
# #print("REF DIR")
# #print(ref_dir)
#
#
# runtime = -1
# cpu = -1
# data = -1
# other = ""
# readme_fnames = [p for p in os.listdir(res_dir) if p.lower().startswith('readme')]
# try:
#     readme_fname = readme_fnames[0]
#     print("Parsing extra information from %s"%readme_fname)
#     with open(os.path.join(input_dir, 'res', readme_fname)) as readme_file:
#         readme = readme_file.readlines()
#         lines = [l.strip() for l in readme if l.find(":")>=0]
#         runtime = float(":".join(lines[0].split(":")[1:]))
#         cpu = int(":".join(lines[1].split(":")[1:]))
#         data = int(":".join(lines[2].split(":")[1:]))
#         other = ":".join(lines[3].split(":")[1:])
# except:
#     print("Error occured while parsing readme.txt")
#     print("Please make sure you have a line for runtime, cpu/gpu, extra data and other (4 lines in total).")
# print("Parsed information:")
# print("Runtime: %f"%runtime)
# print("CPU/GPU: %d"%cpu)
# print("Data: %d"%data)
# print("Other: %s"%other)
#
#
#
#
#
# ref_pngs = sorted([p for p in os.listdir(ref_dir) if p.lower().endswith('png')])
# res_pngs = sorted([p for p in os.listdir(res_dir) if p.lower().endswith('png')])
# if not (len(ref_pngs)==100 and len(res_pngs)==100):
#     raise Exception('Expected 100 .png images, got %d'%len(res_pngs))
#
#
#
#
# scores_psnr = []
# scores_ssim = []
# for (ref_im, res_im) in zip(ref_pngs, res_pngs):
#     print(ref_im,res_im)
#     best_psnr, best_ssim = compute_measures(ref_im, res_im)
#     scores_psnr.append(best_psnr)
#     scores_ssim.append(best_ssim)
#
# psnr = np.mean(scores_psnr)
# mssim = np.mean(scores_ssim)
#
#
#
# # the scores for the leaderboard must be in a file named "scores.txt"
# # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
# with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
#     output_file.write("PSNR:%f\n"%psnr)
#     output_file.write("SSIM:%f\n"%mssim)
#     output_file.write("ExtraRuntime:%f\n"%runtime)
#     output_file.write("ExtraPlatform:%d\n"%cpu)
#     output_file.write("ExtraData:%d\n"%data)
#
# #if __name__ == '__main__':
# #
# #    output_psnr_mse(_open_img(sys.argv[1]),
# #                    _open_img(sys.argv[2]))
#
#
#
#
#
# ##    # unzipped submission data is always in the 'res' subdirectory
# ##    # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
# ##    submission_path = os.path.join(input_dir, 'res', 'answer.txt')
# ##    if not os.path.exists(submission_path):
# ##        sys.exit('Could not find submission file {0}'.format(submission_path))
# ##    with open(submission_path) as submission_file:
# ##        submission = submission_file.read()
# ##
# ##    # unzipped reference data is always in the 'ref' subdirectory
# ##    # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
# ##    with open(os.path.join(input_dir, 'ref', 'truth.txt')) as truth_file:
# ##        truth = truth_file.read()
# ##
# #def _open_img_gray(img_p):
# #    F = scipy.misc.fromimage(Image.open(img_p))
# #    return F
