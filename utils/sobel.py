import numpy as np 
import torch
import cv2 
import torch.nn.functional as F

"""
this is the script to calculate the sobel operator
"""

def sobel_estimator(img, cuda=False, save_figs=False):
    """
    sobel descriptor
    """
    sobel_weight_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_weight_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_weight_45 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    sobel_weight_135 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])
    # process sobel weights
    sobel_weight_x = torch.tensor(sobel_weight_x, dtype=torch.float32, device='cuda' if cuda else 'cpu', requires_grad=False).unsqueeze(0).unsqueeze(0)
    sobel_weight_y = torch.tensor(sobel_weight_y, dtype=torch.float32, device='cuda' if cuda else 'cpu', requires_grad=False).unsqueeze(0).unsqueeze(0)
    sobel_weight_45 = torch.tensor(sobel_weight_45, dtype=torch.float32, device='cuda' if cuda else 'cpu', requires_grad=False).unsqueeze(0).unsqueeze(0)
    sobel_weight_135 = torch.tensor(sobel_weight_135, dtype=torch.float32, device='cuda' if cuda else 'cpu', requires_grad=False).unsqueeze(0).unsqueeze(0)
    # process
    channels = img.size()[1]
    sobel_weight_x = sobel_weight_x.repeat(channels, 1, 1, 1)
    sobel_weight_y = sobel_weight_y.repeat(channels, 1, 1, 1)
    sobel_weight_45 = sobel_weight_45.repeat(channels, 1, 1, 1)
    sobel_weight_135 = sobel_weight_135.repeat(channels, 1, 1, 1)
    # get edge
    edge_x = F.conv2d(img, sobel_weight_x, padding=1, stride=1, groups=channels, bias=None)
    edge_y = F.conv2d(img, sobel_weight_y, padding=1, stride=1, groups=channels, bias=None)
    edge_45 = F.conv2d(img, sobel_weight_45, padding=1, stride=1, groups=channels, bias=None)
    edge_135 = F.conv2d(img, sobel_weight_135, padding=1, stride=1, groups=channels, bias=None)
    if save_figs:
        img_x = edge_x.detach().cpu().numpy()
        img_y = edge_y.detach().cpu().numpy()
        if channels > 1:
            img_x = img_x[0, 0, :, :]
            img_y = img_y[0, 0, :, :]
        else:
            img_x = img_x.squeeze()
            img_y = img_y.squeeze()
        img_x = np.clip(img_x, 0, 1)
        img_y = np.clip(img_y, 0, 1)
        cv2.imwrite('edge_x.png', (img_x * 255).astype(np.uint8))
        cv2.imwrite('edge_y.png', (img_y * 255).astype(np.uint8))
    return edge_x, edge_y, edge_45, edge_135

if __name__ == '__main__':
    img = cv2.imread('img.jpeg',).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32, device='cuda').unsqueeze(0)
    edge_x, edge_y, _, _ = sobel_estimator(img, cuda=True, save_figs=True)