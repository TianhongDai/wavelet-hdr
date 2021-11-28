import numpy as np
import cv2 
import torch
import argparse
from model import Wavelet_UNet
import os
import torch.nn.functional as F
from utils import cal_psnr, load_data, mu_tonemap, radiance_writer
from datetime import datetime
from skimage.metrics import structural_similarity

"""
this script is used to cal the psnr value
"""

parser = argparse.ArgumentParser()
parser.add_argument('--testset-path', type=str, default='./dataset/Kalantari/Test')
parser.add_argument('--cuda', action='store_true', help='use cuda to run the training')
parser.add_argument('--nChannel', type=int, default=6, help='the number of input channels')
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
    scene_dirs = sorted(os.listdir(args.testset_path))
    for scene in scene_dirs:
        im1, im2, im3, ref_hdr = load_data('{}/{}'.format(args.testset_path, scene), use_cuda=args.cuda, image_size=[args.test_h, args.test_w])
        # padding the input image
        im1 = F.pad(im1, (2, 2, 0, 0), mode='reflect')
        im2 = F.pad(im2, (2, 2, 0, 0), mode='reflect')
        im3 = F.pad(im3, (2, 2, 0, 0), mode='reflect')
        with torch.no_grad():
            pred_hdr = net(im1, im2, im3)
            # convert to the numpy array
            pred_hdr = pred_hdr.detach().cpu().numpy().squeeze()
            pred_hdr = np.transpose(pred_hdr, (1, 2, 0))
            # crop the padding
            pred_hdr = pred_hdr[:, 2:-2, :]
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
            radiance_writer('{}/{}.hdr'.format(hdr_path, scene), pred_hdr)
            rgb_im = mu_tonemap(pred_hdr)[:, :, ::-1]
            cv2.imwrite('{}/{}.png'.format(rgb_path, scene), np.uint8(rgb_im * 255))
    print('mean psnr-mu: {:.4f}'.format(np.mean(psnr_mu)))
    print('mean psnr-l: {:.4f}'.format(np.mean(psnr_l)))
    print('mean ssim-mu: {:.4f}'.format(np.mean(ssim_mu)))
    print('mean ssim-l: {:.4f}'.format(np.mean(ssim_l)))