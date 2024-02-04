import numpy as np 
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob

"""
dataset loader for the Kalantari
"""
# this is the gamma correction coefficient
GAMMA = 2.2

class DatasetLoader(Dataset):
    def __init__(self, path, crop_size=256):
        # path
        self.path = path
        self.scene_dirs = sorted(os.listdir(self.path))
        self.crop_size=256

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

    def __getitem__(self, index):
        # list the images in the scence
        scene_dir = '{}/{}'.format(self.path, self.scene_dirs[index])
        in_ldr_paths = sorted(glob(os.path.join(scene_dir, '*.tif')))
        # read the images
        sht = cv2.imread('{}'.format(in_ldr_paths[0]))[:, :, ::-1]
        mid = cv2.imread('{}'.format(in_ldr_paths[1]))[:, :, ::-1]
        lng = cv2.imread('{}'.format(in_ldr_paths[2]))[:, :, ::-1]
        # read the GT
        gt = cv2.imread('{}/HDRImg.hdr'.format(scene_dir), -1).astype(np.float32)
        gt = gt[:, :, ::-1]
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
        sht = sht.astype(np.float32) / 255.0
        mid = mid.astype(np.float32) / 255.0
        lng = lng.astype(np.float32) / 255.0
        # read exposure ratio
        in_exps_path = os.path.join(scene_dir, 'exposure.txt')
        in_exps = np.array(open(in_exps_path).read().split('\n')[:3]).astype(np.float32)
        in_exps -= in_exps.min()
        # conver the image to the linear domain
        sht_hdr = (sht ** GAMMA) / (2 ** in_exps[0])
        mid_hdr = (mid ** GAMMA) / (2 ** in_exps[1])
        lng_hdr = (lng ** GAMMA) / (2 ** in_exps[2])
        # cat and output
        input_ldr = np.concatenate([sht, mid, lng], axis=2)
        input_hdr = np.concatenate([sht_hdr, mid_hdr, lng_hdr], axis=2)
        return input_ldr, input_hdr, gt
    
    def __len__(self):
        return len(self.scene_dirs)

if __name__ == '__main__':
    path = './dataset/Kalantari/Training'
    dataset = DatasetLoader(path)
    train_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=1)
    for i, (in_LDRs, in_HDRs, ref_HDR) in enumerate(train_loader):
        print('{}: {}, {}, {}'.format(i, in_LDRs.shape, in_HDRs.shape, ref_HDR.shape))