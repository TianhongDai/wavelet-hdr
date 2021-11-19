import torch
import torch.nn as nn 
import torch.nn.functional as F
from wavelet_transform.DWT_IDWT_layer import DWT_2D, IDWT_2D
import numpy as np
import argparse

"""
the basic network structure
"""

class AttentionModule(nn.Module):
    """
    attention module
    """
    def __init__(self, nFeat):
        super().__init__()
        self.conv1 = nn.Conv2d(2*nFeat, 2*nFeat, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2*nFeat, nFeat, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, sup, ref):
        x = torch.cat([sup, ref], dim=1)
        x = self.relu(self.conv1(x))
        mask = torch.sigmoid(self.conv2(x))
        sup = sup * mask
        return sup

class Encoder(nn.Module):
    """
    the similar structure as the ECCV paper
    """
    def __init__(self, in_channels, nFeat):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, nFeat, kernel_size=5, stride=1, padding=2)
        self.dwt1 = DWT_2D(wavename='haar')
        self.conv2 = nn.Conv2d(nFeat, nFeat*2, kernel_size=5, stride=1, padding=2)
        self.dwt2 = DWT_2D(wavename='haar')
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(nFeat*2)
    
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        LL1, LH1, HL1, HH1 = self.dwt1(x1)
        x2_ = self.relu(self.bn(self.conv2(LL1)))
        LL2, LH2, HL2, HH2 = self.dwt2(x2_)
        return LL1, LL2, (LH1, HL1, HH1), (LH2, HL2, HH2), x1

class Merge(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.dwt = DWT_2D(wavename='haar')
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        res_layers = []
        for _ in range(num_residual):
            res_layers.append(nn.LeakyReLU())
            res_layers.append(ResidualBlocks(out_channels))
        self.res_blocks = nn.Sequential(*res_layers)
    
    def forward(self, x):
        x1_ = self.relu(self.bn(self.conv(x)))
        LL1, LH1, HL1, HH1 = self.dwt(x1_)
        x2 = self.res_blocks(LL1)
        return LL1, x2, (LH1, HL1, HH1)

class ResidualBlocks(nn.Module):
    def __init__(self, nFeat):
        super().__init__()
        self.conv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(nFeat)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x_res = self.conv1(x)
        x_res = self.bn(self.conv2(self.relu(x_res)))
        return x + x_res

class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels, middle):
        super().__init__()
        self.deconv1 = nn.Conv2d(in_channels, middle, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.Conv2d(middle, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.idwt = IDWT_2D(wavename='haar')
    
    def forward(self, x, x_h):
        x = self.deconv1(x)
        x = self.idwt(x, x_h[0], x_h[1], x_h[2])
        x = self.relu(self.bn(self.deconv2(x)))
        return x

class H_Merge(nn.Module):
    """
    merge the high-frequency components...
    """
    def __init__(self, nChannel):
        super().__init__()
        # lh part
        self.conv_lh_1 = nn.Conv2d(nChannel*3, nChannel, kernel_size=3, stride=1, padding=1)
        self.conv_lh_2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1)
        # hl part
        self.conv_hl_1 = nn.Conv2d(nChannel*3, nChannel, kernel_size=3, stride=1, padding=1)
        self.conv_hl_2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1)
        # hh part
        self.conv_hh_1 = nn.Conv2d(nChannel*3, nChannel, kernel_size=3, stride=1, padding=1)
        self.conv_hh_2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1)
        # relu
        self.relu = nn.ReLU()
    
    def forward(self, h1, h2, h3):
        # merge
        lh_1, hl_1, hh_1 = h1
        lh_2, hl_2, hh_2 = h2
        lh_3, hl_3, hh_3 = h3
        # start to do the fusion
        lh_ = torch.cat([lh_1, lh_2, lh_3], dim=1)
        hl_ = torch.cat([hl_1, hl_2, hl_3], dim=1)
        hh_ = torch.cat([hh_1, hh_2, hh_3], dim=1)
        # start to do the fusion
        lh_ = self.relu(self.conv_lh_1(lh_))
        lh_ = self.conv_lh_2(lh_)
        # 
        hl_ = self.relu(self.conv_hl_1(hl_))
        hl_ = self.conv_hl_2(hl_)
        #
        hh_ = self.relu(self.conv_hh_1(hh_))
        hh_ = self.conv_hh_2(hh_)
        return lh_, hl_, hh_

class Wavelet_UNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        nChannel = args.nChannel
        self.encoder1 = Encoder(nChannel, 64)
        self.encoder2 = Encoder(nChannel, 64)
        self.encoder3 = Encoder(nChannel, 64)
        # define the merge block
        self.merge = Merge(64*2*3, 64*4)
        self.decoder1 = Upsampler(64*4*2, 64*2, 64*4)
        self.decoder2 = Upsampler(64*2*4, 64, 64*2)
        self.decoder3 = Upsampler(64*1*4, 64, 64)
        # high-freq merge
        self.h_merge1 = H_Merge(64*2)
        self.h_merge2 = H_Merge(64*1)
        # conv
        self.conv_hdr = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        # attention
        self.att12 = AttentionModule(128)
        self.att32 = AttentionModule(128)
        # another attention
        self.att11 = AttentionModule(64)
        self.att31 = AttentionModule(64)

    def forward(self, x1, x2, x3):
        x1_1, x1_2, x1_1_h, x1_2_h, _ = self.encoder1(x1)
        x2_1, x2_2, x2_1_h, x2_2_h, x2_global = self.encoder2(x2)
        x3_1, x3_2, x3_1_h, x3_2_h, _ = self.encoder3(x3)
        # do the attention
        x1_1 = self.att11(x1_1, x2_1)
        x3_1 = self.att31(x3_1, x2_1)
        # second attention
        x1_2 = self.att12(x1_2, x2_2)
        x3_2 = self.att32(x3_2, x2_2)
        # merge
        x_ = torch.cat([x1_2, x2_2, x3_2], dim=1)
        xm_1, xm_2, xm_h = self.merge(x_)
        # padding
        d_0 = torch.cat([xm_2, xm_1], dim=1)
        # 64 * 64
        d_1 = self.decoder1(d_0, xm_h)
        # 128 * 128
        d_1 = torch.cat([d_1, x1_2, x2_2, x3_2], dim=1)
        dh_1 = self.h_merge1(x1_2_h, x2_2_h, x3_2_h)
        d_2 = self.decoder2(d_1, dh_1)
        # 256 * 256
        d_2 = torch.cat([d_2, x1_1, x2_1, x3_1], dim=1)
        dh_2 = self.h_merge2(x1_1_h, x2_1_h, x3_1_h)
        d_3 = self.decoder3(d_2, dh_2)
        d_3 = d_3 + x2_global
        out = self.conv_hdr(d_3)
        # output
        return torch.sigmoid(out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nChannel', type=int, default=6, help='the number of input channels')
    args = parser.parse_args()
    # define the wavelet
    net = Wavelet_UNet(args)
    net.cuda()
    inputs = np.ones((1, 6, 256, 256), dtype=np.float32)
    inputs = torch.tensor(inputs, dtype=torch.float32, device='cuda')
    with torch.no_grad():
        outputs = net(inputs, inputs, inputs)
    print(outputs.shape)