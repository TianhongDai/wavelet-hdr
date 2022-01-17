import numpy as np 
import cv2 
import torch
import os
from glob import glob
import math
import json

"""
this script is used to contain most useful functions
"""

GAMMA = 2.2

def cal_gamma(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    gray = r*0.2989 + g*0.5870 + b*0.1140
    gray_mean = np.mean(gray)
    gamma = np.min([np.log(0.5) / np.log(gray_mean), 0.45])
    return gamma

def apply_gamma(img, gamma):
    rgb = np.power(np.minimum(img, 1.0), gamma)
    return rgb

def dirtDM(img):
    c_R = img[:, :, 0:1]
    c_RG = img[:, :, 1:2]
    c_B = img[:, :, 2:3]
    c_BG = img[:, :, 3:]
    RGB = np.concatenate([c_R, (c_RG + c_BG) / 2.0, c_B], axis=2)
    return RGB

def simple_isp(im):
    #im = create_rawRGGB(im)
    im = dirtDM(im)
    im = np.clip(im / im.max(), 0, 1)
    im = apply_gamma(im, cal_gamma(im))
    return np.uint8(im*255)[:, :, ::-1]

def cal_er_explict(exif_s, exif_l):
    if(exif_l['aperture']==exif_s['aperture']):
        er = (exif_l['iso']*exif_l['exposure']) / (exif_s['iso']*exif_s['exposure'])
    else:
        er = (exif_l['iso']*exif_l['exposure']) / (exif_s['iso']*exif_s['exposure']) * (2 ** (exif_l['aperture'] / exif_s['aperture']))
    return er

def create_rawRGGB(raw):
    raw = raw[:, :, np.newaxis]
    c_R = raw[0::2, 0::2]
    c_RG = raw[0::2, 1::2]
    c_BG = raw[1::2, 0::2]
    c_B = raw[1::2, 1::2]
    rawRGB = np.concatenate([c_R, c_RG, c_B, c_BG], axis=2)
    return rawRGB

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
    sht = np.fromfile('{}/im1.raw'.format(path), np.uint16).reshape(1732*2, 2608*2)
    sht = sht.astype(np.float32) / (2**14 - 1)
    mid = np.fromfile('{}/im2.raw'.format(path), np.uint16).reshape(1732*2, 2608*2)
    mid = mid.astype(np.float32) / (2**14 - 1)
    lng = np.fromfile('{}/im3.raw'.format(path), np.uint16).reshape(1732*2, 2608*2)
    lng = lng.astype(np.float32) / (2**14 - 1)
    # read the GT
    gt = np.fromfile('{}/gt.raw'.format(path), np.float32).reshape(1732*2, 2608*2)
    # convert into rggb
    sht = create_rawRGGB(sht) 
    mid = create_rawRGGB(mid) 
    lng = create_rawRGGB(lng) 
    gt = create_rawRGGB(gt)
    # read exposure ratio
    with open('{}/im1.txt'.format(path), 'r') as f1:
        exif1 = json.load(f1)
    with open('{}/im2.txt'.format(path), 'r') as f2:
        exif2 = json.load(f2)
    with open('{}/im3.txt'.format(path), 'r') as f3:
        exif3 = json.load(f3)
    # exif 
    er_mid = cal_er_explict(exif1, exif2)
    er_lng = cal_er_explict(exif1, exif3)
    # normalize
    sht_hdr = sht.copy()
    mid_hdr = mid / er_mid
    lng_hdr = lng / er_lng
    # normalize
    sht = sht ** (1 / 2.2)
    mid = mid ** (1 / 2.2)
    lng = lng ** (1 / 2.2)
    # gamma correction of the input ldr images 
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

def radiance_writer(out_path, image):
    """
    write the hdr image
    """
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))
        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)
        rgbe.flatten().tofile(f)

if __name__ == '__main__':
    pass
