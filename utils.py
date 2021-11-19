import numpy as np 
import cv2 
import torch
import os
from glob import glob
import math

"""
this script is used to contain most useful functions
"""

GAMMA = 2.2

def cal_psnr(im1, im2):
    mse = np.mean((im1 - im2)**2)
    if mse == 0:
        return 100
    pixel_max = 1
    return 20 * math.log10(pixel_max / math.sqrt(mse))

def mu_tonemap(x, mu=5000):
    return np.log(1 + mu * x) / np.log(1 + mu)

def center_crop(x, image_size=[1000, 1496]):
    crop_h, crop_w = image_size
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    #return cv2.resize(x[max(0,j):min(h,j+crop_h), max(0,i):min(w,i+crop_w)], (crop_w, crop_h))
    return x[max(0,j):min(h,j+crop_h), max(0,i):min(w,i+crop_w)]

def load_data(path, use_cuda, image_size):
    in_ldr_paths = sorted(glob(os.path.join(path, '*.tif')))
    sht = cv2.imread('{}'.format(in_ldr_paths[0]))[:, :, ::-1]
    mid = cv2.imread('{}'.format(in_ldr_paths[1]))[:, :, ::-1]
    lng = cv2.imread('{}'.format(in_ldr_paths[2]))[:, :, ::-1]
    # noramlize
    sht = sht.astype(np.float32) / 255.0
    mid = mid.astype(np.float32) / 255.0
    lng = lng.astype(np.float32) / 255.0
    # read the ground truth
    gt = cv2.imread('{}/HDRImg.hdr'.format(path), -1).astype(np.float32)
    gt = gt[:, :, ::-1]
    # crop
    """
    sht = center_crop(sht, image_size=image_size)
    mid = center_crop(mid, image_size=image_size)
    lng = center_crop(lng, image_size=image_size)
    gt = center_crop(gt, image_size=image_size)
    """
    # gamma correction of the input ldr images 
    in_exps = np.array(open('{}/exposure.txt'.format(path)).read().split('\n')[:3]).astype(np.float32)
    in_exps -= in_exps.min()
    # conver the image to the linear domain
    sht_hdr = (sht ** GAMMA) / (2 ** in_exps[0])
    mid_hdr = (mid ** GAMMA) / (2 ** in_exps[1])
    lng_hdr = (lng ** GAMMA) / (2 ** in_exps[2])
    # create tensors
    im1 = np.concatenate([sht, sht_hdr], axis=2)
    im2 = np.concatenate([mid, mid_hdr], axis=2)
    im3 = np.concatenate([lng, lng_hdr], axis=2)
    # transpose 
    im1 = np.transpose(im1, (2, 0, 1))
    im2 = np.transpose(im2, (2, 0, 1))
    im3 = np.transpose(im3, (2, 0, 1))
    # into tensor
    im1 = torch.tensor(im1, dtype=torch.float32, device='cuda' if use_cuda else 'cpu').unsqueeze(0)
    im2 = torch.tensor(im2, dtype=torch.float32, device='cuda' if use_cuda else 'cpu').unsqueeze(0)
    im3 = torch.tensor(im3, dtype=torch.float32, device='cuda' if use_cuda else 'cpu').unsqueeze(0)
    return im1, im2, im3, gt

if __name__ == '__main__':
    pass