import numpy as np
import cv2 
import torch
import argparse
from model import Wavelet_UNet
import os
import torch.nn.functional as F
from utils import cal_psnr, load_data, mu_tonemap
from datetime import datetime

"""
this script is used to cal the psnr value
"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default='./dataset/Kalantari/Training')
parser.add_argument('--testset-path', type=str, default='./dataset/Kalantari/Test')
parser.add_argument('--cuda', action='store_true', help='use cuda to run the training')
parser.add_argument('--batch-size', type=int, default=16, help='the batch size')
parser.add_argument('--nChannel', type=int, default=6, help='the number of input channels')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--epochs', type=int, default=20000, help='total epochs')
parser.add_argument('--test-w', type=int, default=1496, help='width')
parser.add_argument('--test-h', type=int, default=1000, help='height')
parser.add_argument('--save-interval', type=int, default=10)
parser.add_argument('--use-parallel', action='store_true', help='if use multiple-gpu run this')
parser.add_argument('--save-path', type=str, default='saved_models')

if __name__ == '__main__':
    # get the args
    args = parser.parse_args()
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
    psnr_mu = []
    scene_dirs = sorted(os.listdir(args.testset_path))
    for scene in scene_dirs:
        im1, im2, im3, ref_hdr = load_data('{}/{}'.format(args.testset_path, scene), use_cuda=args.cuda, image_size=[args.test_h, args.test_w])
        im1 = F.pad(im1, (2, 2, 0, 0), mode='reflect')
        im2 = F.pad(im2, (2, 2, 0, 0), mode='reflect')
        im3 = F.pad(im3, (2, 2, 0, 0), mode='reflect')
        with torch.no_grad():
            pred_hdr = net(im1, im2, im3)
            # padd the inputs
            # convert to the numpy array
            pred_hdr = pred_hdr.detach().cpu().numpy().squeeze()
            pred_hdr = np.transpose(pred_hdr, (1, 2, 0))
            # crop the padding
            pred_hdr = pred_hdr[:, 2:-2, :]
        # tonemapping and calculate the psnr
        psnr_mu_ = cal_psnr(mu_tonemap(pred_hdr), mu_tonemap(ref_hdr))
        print('[{}] scene_id: {}, psnr_mu: {:.3f}'.format(datetime.now(), scene, psnr_mu_))
        psnr_mu.append(psnr_mu_)
    print('mean psnr-mu: {:.4f}'.format(np.mean(psnr_mu)))
        