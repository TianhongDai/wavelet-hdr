import numpy as np 
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import json

"""
dataset loader for the Kalantari
"""
# this is the gamma correction coefficient
GAMMA = 2.2

class DatasetLoader(Dataset):
    def __init__(self, path, crop_size=256, mode='train'):
        # path
        self.path = path
        # read the test data
        test_scenes = []
        with open('test.lst', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                test_scenes.append(line)
        self.scene_dirs = sorted(os.listdir(self.path))
        if mode == 'train':
            self.scene_dirs_ = list(set(self.scene_dirs) - set(test_scenes))
        elif mode == 'test':
            self.scene_dirs_ = test_scenes
        self.crop_size=256

    def _create_rawRGGB(self, raw):
        raw = raw[:, :, np.newaxis]
        c_R = raw[0::2, 0::2]
        c_RG = raw[0::2, 1::2]
        c_BG = raw[1::2, 0::2]
        c_B = raw[1::2, 1::2]
        rawRGB = np.concatenate([c_R, c_RG, c_B, c_BG], axis=2)
        return rawRGB
    
    def _random_crop(self, img, x, y, crop_size=256):
        """
        random crop the whole image into patches
        """
        img = img[x:x+crop_size, y:y+crop_size, :].copy()
        return img

    def _random_flip(self, img, mode='v'):
        """
        random flip the whole image in the vertical / horizontal dir
        """
        if mode == 'v':
            return np.flip(img, 0)
        else:
            return np.flip(img, 1)

    def _random_rot(self, img, rot_coin):
        img = np.rot90(img, rot_coin).copy()
        return img

    def cal_er_explict(self, exif_s, exif_l):
        if(exif_l['aperture']==exif_s['aperture']):
            er = (exif_l['iso']*exif_l['exposure']) / (exif_s['iso']*exif_s['exposure'])
        else:
            er = (exif_l['iso']*exif_l['exposure']) / (exif_s['iso']*exif_s['exposure']) * (2 ** (exif_l['aperture'] / exif_s['aperture']))
        return er

    def __getitem__(self, index):
        # list the images in the scence
        scene_dir = '{}/{}'.format(self.path, self.scene_dirs_[index])
        # read the images
        sht = np.fromfile('{}/im1.raw'.format(scene_dir), np.uint16).reshape(1732*2, 2608*2)
        sht = sht.astype(np.float32) / (2**14 - 1)
        mid = np.fromfile('{}/im2.raw'.format(scene_dir), np.uint16).reshape(1732*2, 2608*2)
        mid = mid.astype(np.float32) / (2**14 - 1)
        lng = np.fromfile('{}/im3.raw'.format(scene_dir), np.uint16).reshape(1732*2, 2608*2)
        lng = lng.astype(np.float32) / (2**14 - 1)
        # read the GT
        gt = np.fromfile('{}/gt.raw'.format(scene_dir), np.float32).reshape(1732*2, 2608*2)
        # convert into rggb
        sht = self._create_rawRGGB(sht) 
        mid = self._create_rawRGGB(mid) 
        lng = self._create_rawRGGB(lng) 
        gt = self._create_rawRGGB(gt)
        # process the data
        h, w, _ = sht.shape
        # 1. random crop the image into patchs 
        x = np.random.randint(0, h - self.crop_size - 1)
        y = np.random.randint(0, w - self.crop_size - 1)
        # start the crop
        sht = self._random_crop(sht, x, y, self.crop_size)
        mid = self._random_crop(mid, x, y, self.crop_size)
        lng = self._random_crop(lng, x, y, self.crop_size)
        gt = self._random_crop(gt, x, y, self.crop_size)
        # 2. random flip
        flip_coin = np.random.random()
        if flip_coin > 0.5:
            mode = 'v' if np.random.random() > 0.5 else 'h'
            sht = self._random_flip(sht, mode=mode)
            mid = self._random_flip(mid, mode=mode)
            lng = self._random_flip(lng, mode=mode)
            gt = self._random_flip(gt, mode=mode) 
        # 3. random rot
        rot_coin = np.random.randint(0, 4)
        sht = self._random_rot(sht, rot_coin)
        mid = self._random_rot(mid, rot_coin)
        lng = self._random_rot(lng, rot_coin)
        gt = self._random_rot(gt, rot_coin)
        # start to process the inputs, normalize and change the color channel to rgb instead of bgr
        # read exposure ratio
        with open('{}/im1.txt'.format(scene_dir), 'r') as f1:
            exif1 = json.load(f1)
        with open('{}/im2.txt'.format(scene_dir), 'r') as f2:
            exif2 = json.load(f2)
        with open('{}/im3.txt'.format(scene_dir), 'r') as f3:
            exif3 = json.load(f3)
        # exif 
        er_mid = self.cal_er_explict(exif1, exif2)
        er_lng = self.cal_er_explict(exif1, exif3)
        # normalize
        sht_hdr = sht.copy()
        mid_hdr = mid / er_mid
        lng_hdr = lng / er_lng
        # normalize
        sht = sht ** (1 / 2.2)
        mid = mid ** (1 / 2.2)
        lng = lng ** (1 / 2.2)
        # cat and output
        input_ldr = np.concatenate([sht, mid, lng], axis=2)
        input_hdr = np.concatenate([sht_hdr, mid_hdr, lng_hdr], axis=2)
        return input_ldr, input_hdr, gt
    
    def __len__(self):
        return len(self.scene_dirs_)

if __name__ == '__main__':
    path = '../../dataset/raw_dataset'
    dataset = DatasetLoader(path)
    _, _, _ = dataset[0] 
    train_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, num_workers=4)
    for i, (in_LDRs, in_HDRs, ref_HDR) in enumerate(train_loader):
        print('{}: {}, {}, {}'.format(i, in_LDRs.shape, in_HDRs.shape, ref_HDR.shape))