import numpy as np 
import torch
import os
import argparse
from model import Wavelet_UNet
import cv2
from utils import cal_psnr, load_data, mu_tonemap, create_rawRGGB, cal_er_explict
import torch.nn.functional as F

"""
this script is used to evaluate the network
"""
def eval_network(args, net):
    # loaad the dataset
    psnr_mu = []
    test_scenes = []
    with open('test.lst', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            test_scenes.append(line)
    for scene in test_scenes:
        im1, im2, im3, ref_hdr = load_data('{}/{}'.format(args.testset_path, scene), use_cuda=args.cuda, image_size=[args.test_h, args.test_w])
        # pad the image
        """
        1 2 3 4
        5 6 7 8
        """
        im1 = im1[:, :, 66:-66, 24:-24]
        im2 = im2[:, :, 66:-66, 24:-24]
        im3 = im3[:, :, 66:-66, 24:-24]
        ref_hdr = ref_hdr[66:-66, 24:-24, :]
        #pad
        #im1 = F.pad(im1, (0, 0, 6, 6), mode='reflect')
        #im2 = F.pad(im2, (0, 0, 6, 6), mode='reflect')
        #im3 = F.pad(im3, (0, 0, 6, 6), mode='reflect')
        # block 1
        im1_1 = im1[:, :, :800, :640]
        im2_1 = im2[:, :, :800, :640]
        im3_1 = im3[:, :, :800, :640]
        # block2
        im1_2 = im1[:, :, :800, 640:1280]
        im2_2 = im2[:, :, :800, 640:1280]
        im3_2 = im3[:, :, :800, 640:1280]
        # block3
        im1_3 = im1[:, :, :800, 1280:1920]
        im2_3 = im2[:, :, :800, 1280:1920]
        im3_3 = im3[:, :, :800, 1280:1920]
        # block 4
        im1_4 = im1[:, :, :800, 1920:]
        im2_4 = im2[:, :, :800, 1920:]
        im3_4 = im3[:, :, :800, 1920:]
        # block 5
        im1_5 = im1[:, :, 800:, :640]
        im2_5 = im2[:, :, 800:, :640]
        im3_5 = im3[:, :, 800:, :640]
        # block6
        im1_6 = im1[:, :, 800:, 640:1280]
        im2_6 = im2[:, :, 800:, 640:1280]
        im3_6 = im3[:, :, 800:, 640:1280]
        # block7
        im1_7 = im1[:, :, 800:, 1280:1920]
        im2_7 = im2[:, :, 800:, 1280:1920]
        im3_7 = im3[:, :, 800:, 1280:1920]
        # block 8
        im1_8 = im1[:, :, 800:, 1920:]
        im2_8 = im2[:, :, 800:, 1920:]
        im3_8 = im3[:, :, 800:, 1920:]
        with torch.no_grad():
            pred_hdr1 = net(im1_1, im2_1, im3_1)
            pred_hdr2 = net(im1_2, im2_2, im3_2)
            pred_hdr3 = net(im1_3, im2_3, im3_3)
            pred_hdr4 = net(im1_4, im2_4, im3_4)
            pred_hdr5 = net(im1_5, im2_5, im3_5)
            pred_hdr6 = net(im1_6, im2_6, im3_6)
            pred_hdr7 = net(im1_7, im2_7, im3_7)
            pred_hdr8 = net(im1_8, im2_8, im3_8)
            # convert to the numpy array
            pred_hdr1 = pred_hdr1.detach().cpu().numpy().squeeze()
            pred_hdr2 = pred_hdr2.detach().cpu().numpy().squeeze()
            pred_hdr3 = pred_hdr3.detach().cpu().numpy().squeeze()
            pred_hdr4 = pred_hdr4.detach().cpu().numpy().squeeze()
            pred_hdr5 = pred_hdr5.detach().cpu().numpy().squeeze()
            pred_hdr6 = pred_hdr6.detach().cpu().numpy().squeeze()
            pred_hdr7 = pred_hdr7.detach().cpu().numpy().squeeze()
            pred_hdr8 = pred_hdr8.detach().cpu().numpy().squeeze()
            # transpose
            pred_hdr1 = np.transpose(pred_hdr1, (1, 2, 0))
            pred_hdr2 = np.transpose(pred_hdr2, (1, 2, 0))
            pred_hdr3 = np.transpose(pred_hdr3, (1, 2, 0))
            pred_hdr4 = np.transpose(pred_hdr4, (1, 2, 0))
            pred_hdr5 = np.transpose(pred_hdr5, (1, 2, 0))
            pred_hdr6 = np.transpose(pred_hdr6, (1, 2, 0))
            pred_hdr7 = np.transpose(pred_hdr7, (1, 2, 0))
            pred_hdr8 = np.transpose(pred_hdr8, (1, 2, 0))
            pred_up = np.concatenate([pred_hdr1, pred_hdr2, pred_hdr3, pred_hdr4], axis=1)
            pred_down = np.concatenate([pred_hdr5, pred_hdr6, pred_hdr7, pred_hdr8], axis=1)
            pred_hdr = np.concatenate([pred_up, pred_down], axis=0)
        # tonemapping and calculate the psnr
        psnr_mu_ = cal_psnr(mu_tonemap(pred_hdr), mu_tonemap(ref_hdr))
        psnr_mu.append(psnr_mu_)
    return np.mean(psnr_mu)

if __name__ == '__main__':
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
    # get the argument
    args = parser.parse_args()
    net = Wavelet_UNet(args)
    net.cuda()
    mean_psnr_mu = eval_network(args, net)
    print(mean_psnr_mu)