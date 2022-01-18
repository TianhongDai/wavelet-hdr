import numpy as np
import cv2 
import torch
import argparse
from model import Wavelet_UNet
import os
import torch.nn.functional as F
from utils import cal_psnr, load_data, mu_tonemap, radiance_writer, simple_isp, create_rawRGGB, dirtDM
from datetime import datetime
from skimage.metrics import structural_similarity

"""
this script is used to cal the psnr value
"""

parser = argparse.ArgumentParser()
parser.add_argument('--testset-path', type=str, default='./dataset/Kalantari/Test')
parser.add_argument('--cuda', action='store_true', help='use cuda to run the training')
parser.add_argument('--nChannel', type=int, default=8, help='the number of input channels')
parser.add_argument('--test-w', type=int, default=1500, help='width')
parser.add_argument('--test-h', type=int, default=1000, help='height')
parser.add_argument('--save-path', type=str, default='saved_models')
parser.add_argument('--save-image', action='store_true', help='save the processed hdr image')
parser.add_argument('--save-hdr-path', type=str, default='./results')

if __name__ == '__main__':
    # get the args
    args = parser.parse_args()
    # build the dir to store hdr and rgb image
    if args.save_image:
        hdr_path = '{}/hdr'.format(args.save_hdr_path)
        rgb_path = '{}/rgb'.format(args.save_hdr_path)
        if not os.path.exists(hdr_path):
            os.makedirs(hdr_path, exist_ok=True)
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path, exist_ok=True)
    # build the model
    net = Wavelet_UNet(args)
    # lado the model 
    net_param, _ = torch.load('{}/model.pt'.format(args.save_path), map_location='cpu')
    net.load_state_dict(net_param)
    if args.cuda:
        net.cuda()
    # eval
    net.eval()
    # start to load the data
    psnr_mu, psnr_l, ssim_mu, ssim_l = [], [], [], []
    #scene_dirs = sorted(os.listdir(args.testset_path))
    test_scenes = []
    with open('test.lst', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            test_scenes.append(line)
    for scene in test_scenes:
        im1, im2, im3, ref_hdr = load_data('{}/{}'.format(args.testset_path, scene), use_cuda=args.cuda, image_size=[args.test_h, args.test_w])
        # padding the input image
        im1 = im1[:, :, 66:-66, 24:-24]
        im2 = im2[:, :, 66:-66, 24:-24]
        im3 = im3[:, :, 66:-66, 24:-24]
        ref_hdr = ref_hdr[66:-66, 24:-24, :]
        with torch.no_grad():
            # block1
            im1_1 = im1[:, :, :800, :1280]
            im2_1 = im2[:, :, :800, :1280]
            im3_1 = im3[:, :, :800, :1280]
            # block2
            im1_2 = im1[:, :, :800, 1280:]
            im2_2 = im2[:, :, :800, 1280:]
            im3_2 = im3[:, :, :800, 1280:]
            # block3
            im1_3 = im1[:, :, 800:, :1280]
            im2_3 = im2[:, :, 800:, :1280]
            im3_3 = im3[:, :, 800:, :1280]
            # block 4
            im1_4 = im1[:, :, 800:, 1280:]
            im2_4 = im2[:, :, 800:, 1280:]
            im3_4 = im3[:, :, 800:, 1280:]
            # pread
            pred_hdr1 = net(im1_1, im2_1, im3_1)
            pred_hdr2 = net(im1_2, im2_2, im3_2)
            pred_hdr3 = net(im1_3, im2_3, im3_3)
            pred_hdr4 = net(im1_4, im2_4, im3_4)
            # convert to the numpy array
            pred_hdr1 = pred_hdr1.detach().cpu().numpy().squeeze()
            pred_hdr2 = pred_hdr2.detach().cpu().numpy().squeeze()
            pred_hdr3 = pred_hdr3.detach().cpu().numpy().squeeze()
            pred_hdr4 = pred_hdr4.detach().cpu().numpy().squeeze()
            # transpose
            pred_hdr1 = np.transpose(pred_hdr1, (1, 2, 0))
            pred_hdr2 = np.transpose(pred_hdr2, (1, 2, 0))
            pred_hdr3 = np.transpose(pred_hdr3, (1, 2, 0))
            pred_hdr4 = np.transpose(pred_hdr4, (1, 2, 0))
            # crop the padding
            pred_up = np.concatenate([pred_hdr1, pred_hdr2], axis=1)
            pred_down = np.concatenate([pred_hdr3, pred_hdr4], axis=1)
            pred_hdr = np.concatenate([pred_up, pred_down], axis=0)
        # tonemapping and calculate the psnr
        psnr_mu_ = cal_psnr(mu_tonemap(pred_hdr), mu_tonemap(ref_hdr))
        psnr_l_ = cal_psnr(pred_hdr, ref_hdr)
        # calculate ssim 
        ssim_mu_ = structural_similarity(mu_tonemap(pred_hdr), mu_tonemap(ref_hdr), multichannel=True)
        ssim_l_ = structural_similarity(pred_hdr, ref_hdr, multichannel=True)
        print('[{}] scene_id: {}, psnr_mu: {:.3f}'.format(datetime.now(), scene, psnr_mu_))
        # collect the metrics
        psnr_mu.append(psnr_mu_)
        psnr_l.append(psnr_l_)
        ssim_mu.append(ssim_mu_)
        ssim_l.append(ssim_l_)
        if args.save_image:
            hdr = dirtDM(pred_hdr)
            radiance_writer('{}/{}.hdr'.format(hdr_path, scene), hdr)
            rgb_im = simple_isp(pred_hdr)
            cv2.imwrite('{}/{}.png'.format(rgb_path, scene), rgb_im)
    print('mean psnr-mu: {:.4f}'.format(np.mean(psnr_mu)))
    print('mean psnr-l: {:.4f}'.format(np.mean(psnr_l)))
    print('mean ssim-mu: {:.4f}'.format(np.mean(ssim_mu)))
    print('mean ssim-l: {:.4f}'.format(np.mean(ssim_l)))