import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from torch.autograd import Variable

DEVICE = 'cuda'

#code heavily inspired by PWCNet
def predict_with_warp_flo(args):

    # compile list
    images = glob.glob(os.path.join(args.path, '*.png')) + \
        glob.glob(os.path.join(args.path, '*.jpg'))
    images = sorted(images)
    img_list = []
    for imgfile in images[::-1][:-1]:
        img = np.array(Image.open(imgfile)).astype(np.uint8)
        img = img / 255
        img = np.moveaxis(img, -1, 0)
        img_list.append(img.tolist())    
    x = torch.FloatTensor(img_list).to(DEVICE) #(N, 3, H, W)

    masks = glob.glob(os.path.join(args.mask, '*.png')) + \
        glob.glob(os.path.join(args.mask, '*.jpg'))
    masks = sorted(masks)
    mask_list = []
    for imgfile in masks[:-1][::-1]:
        img = np.array(Image.open(imgfile)).astype(np.uint8)
        img = np.expand_dims(img, axis=2)
        img = img / 255
        img = np.moveaxis(img, -1, 0)
        mask_list.append(img.tolist())
    masks = torch.FloatTensor(mask_list).to(DEVICE) #(N, 3, H, W)

    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()


    flo =  np.load(args.flow)
    flo = torch.FloatTensor(flo).permute(0,3,1,2)
    vgrid = Variable(grid) +2* flo
    
    vgrid = vgrid.to(DEVICE)

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :].clone() / max(W-1, 1)-1.0
    vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :].clone() / max(H-1, 1)-1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output  = F.grid_sample(x, vgrid, align_corners=True)
    
    #mask = torch.autograd.Variable(torch.ones(x.size())).to(DEVICE)
    
    #mask = F.grid_sample(mask, vgrid, align_corners=True)

    #mask[mask < 0.9999] = 0
    #mask[mask > 0] = 1
    
    # output = output * masks
    # mask =  mask.cpu().data.numpy()    
    
    output = output.permute(0,2,3,1)
    output = output.cpu().data.numpy()
    x = x.permute(0, 2, 3, 1).cpu().numpy()

    size = output.shape[0]
    for i in range(size):
        plt.imsave("/phoenix/S3/alh293/data/raft_data/outputs/"+args.name+"/img_" + str(i+1) + ".jpg",output[i])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--flow', help="flo of dataset")
    parser.add_argument('--name', help='name of save folder')
    parser.add_argument('--mask', help='mask folder')
    args = parser.parse_args()
  
    predict_with_warp_flo(args)


