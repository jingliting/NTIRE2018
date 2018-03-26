import tensorlayer as tl
import os


def data_for_predict(input_dir=None, groundtruth_dir=None, postfix_len=2):
    lr_list = sorted(os.listdir(input_dir))
    hr_list = get_hrimg_list(lr_list, postfix_len)
    lr_imgs = tl.visualize.read_images(lr_list, input_dir)
    if groundtruth_dir:
        groundtruth_imgs = tl.visualize.read_images(hr_list, groundtruth_dir)
    else:
        groundtruth_imgs=None
    return lr_list, lr_imgs, groundtruth_imgs


def get_hrimg_list(lr_list, postfix_len):
    """
    Get HR image name list according to LR image name list
    """
    image_hr_list= []
    for i in range(len(lr_list)):
        name_lr,postfix = os.path.splitext(lr_list[i])
        name_hr = name_lr[:-1*postfix_len]
        image_hr_list.append(name_hr+postfix)
    return image_hr_list