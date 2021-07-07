import sys

sys.path.append("core")

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

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def get_images(path) -> torch.FloatTensor:
    images = glob.glob(os.path.join(f"{path}", "*.png")) + glob.glob(
        os.path.join(f"{path}", "*.jpg")
    )
    images = sorted(images)
    img_list = []
    for imgfile in images[1:]:
        img = np.array(Image.open(imgfile).resize((608, 1080))).astype(np.uint8)
        img = np.moveaxis(img, -1, 0)
        img_list.append(img.tolist())
    x = torch.FloatTensor(img_list).to(DEVICE)  # (N, 3, H, W)

    return x.detach().numpy().astype(np.uint8)


def batch_backwards_warp(args) -> None:
    imgs = get_images(args.path)
    flow = np.load(args.flow)

    B, C, H, W = imgs.shape

    # build grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    # update flow and normalise to range [-1,1]
    vgrid = Variable(grid) + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(torch.FloatTensor(imgs), vgrid, align_corners=True)

    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    for i, warped_img in enumerate(output.detach().cpu().numpy().astype(np.uint8)):
        plt.imshow(warped_img.transpose(1, 2, 0))
        plt.savefig(f"{args.save_path}/img_{i+1}.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="dataset for evaluation")
    parser.add_argument("--flow", help="flow of dataset")
    parser.add_argument("--save_path", help="name of save folder")
    parser.add_argument("--mask", help="mask folder", required=False)
    args = parser.parse_args()

    batch_backwards_warp(args)
