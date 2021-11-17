import numpy as np 
import torch
import argparse
from model import Wavelet_UNet
from dataset_loader import DatasetLoader
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sobel import sobel_estimator
from datetime import datetime
import os
from eval import eval_network
"""
this is the script to train the network
"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default='./dataset/Kalantari/Training')
parser.add_argument('--testset-path', type=str, default='./dataset/Kalantari/Test')
parser.add_argument('--cuda', action='store_true', help='use cuda to run the training')
parser.add_argument('--batch-size', type=int, default=16, help='the batch size')
parser.add_argument('--nChannel', type=int, default=6, help='the number of input channels')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--epochs', type=int, default=20000, help='total epochs')
parser.add_argument('--test-w', type=int, default=1496, help='width')
parser.add_argument('--test-h', type=int, default=1000, help='height')
parser.add_argument('--save-interval', type=int, default=10)
parser.add_argument('--use-parallel', action='store_true', help='if use multiple-gpu run this')
parser.add_argument('--save-path', type=str, default='saved_models')


def mu_tonemap(x, mu=5000):
    return torch.log(1 + mu * x) / np.log(1 + mu)

def process_tensors(in_ldrs, in_hdrs, ref_hdr, use_cuda):
    """
    convert the tensor
    """
    # process the input
    im1 = torch.cat([in_ldrs[:, :, :, :3], in_hdrs[:, :, :, :3]], dim=3).permute(0, 3, 1, 2)
    im2 = torch.cat([in_ldrs[:, :, :, 3:6], in_hdrs[:, :, :, 3:6]], dim=3).permute(0, 3, 1, 2)
    im3 = torch.cat([in_ldrs[:, :, :, 6:9], in_hdrs[:, :, :, 6:9]], dim=3).permute(0, 3, 1, 2)
    # process the ground truth
    ref_hdr = ref_hdr.permute(0, 3, 1, 2)
    # send the tensor into the cuda
    im1 = im1.to('cuda' if use_cuda else 'cpu')
    im2 = im2.to('cuda' if use_cuda else 'cpu')
    im3 = im3.to('cuda' if use_cuda else 'cpu')
    ref_hdr = ref_hdr.to('cuda' if use_cuda else 'cpu')
    return im1, im2, im3, ref_hdr

if __name__ == '__main__':
    # get the arguments
    args = parser.parse_args()
    # creat the save dir
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    # get the dataset
    dataset = DatasetLoader(args.dataset_path)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # define the network
    net = Wavelet_UNet(args)
    if args.use_parallel:
        net = nn.DataParallel(net)
    # if use cuda
    if args.cuda:
        net.cuda()
    # define the optimizer
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # start the training
    net.train()
    best_psnr_mu = 0
    for epoch_id in range(args.epochs):
        for i, (in_ldrs, in_hdrs, ref_hdr) in enumerate(train_loader):
            im1, im2, im3, ref_hdr = process_tensors(in_ldrs, in_hdrs, ref_hdr, use_cuda=args.cuda)
            # tonemapping and calculate the loss
            generate_hdr = net(im1, im2, im3)
            # tone mapping
            generate_hdr_tonemap = mu_tonemap(generate_hdr)
            ref_hdr_tonemap = mu_tonemap(ref_hdr)
            loss = torch.abs(generate_hdr_tonemap - ref_hdr_tonemap).mean()
            # get the sobel loss - currently we use the normal sobel
            generate_grad_x, generate_grad_y, generate_grad_45, generate_grad_135 = sobel_estimator(generate_hdr_tonemap, cuda=args.cuda)
            ref_grad_x, ref_grad_y, ref_grad_45, ref_grad_135 = sobel_estimator(ref_hdr_tonemap, cuda=args.cuda)
            sobel_loss = 0.25 * torch.abs(generate_grad_x - ref_grad_x).mean() + 0.25 * torch.abs(generate_grad_y - ref_grad_y).mean()
            # the total loss
            loss = loss + sobel_loss
            # start to update the network
            optim.zero_grad()
            loss.backward()
            optim.step()
        print('[{}] epoch: {}, iteration: {}/{}, loss:{:.3f}, best_psnr: {:.3f}'.format(datetime.now(), epoch_id, i+1, len(train_loader), loss.item(), best_psnr_mu))
        if epoch_id % 10 == 0:
            # eval the model
            print('[{}] start to evaluate the model'.format(datetime.now()))
            net.eval()
            cur_psnr_mu = eval_network(args, net)
            if cur_psnr_mu > best_psnr_mu:
                best_psnr_mu = cur_psnr_mu
                torch.save([net.state_dict(), optim.state_dict()], '{}/model.pt'.format(args.save_path))