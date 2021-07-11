import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import imageio

"""
This file contains all loss functions that are "unsupervised" - i.e., they will take in only one tensor ideally,
which should be a tensor containing n image samples. See specific examples and docstrings.
"""

i = 1


def l1_loss(results):
    """
    Basic temporal loss. Computes L1 distance between result[i] and result[i+1].
    """
    losses = results[1:] - results[:-1]

    return losses.sum()


def l1_reconstruction_loss(predicted, gt):
    return torch.abs(predicted - gt).sum()


def l2_mse_loss(predicted, gt):
    criterion = torch.nn.MSELoss()
    loss = criterion(predicted, gt)
    return loss


def optical_flow_loss(alb1, alb2, mask1, flow, device):
    """
    Computes the MSE between the albedo of 2 images. Uses optical flow correspondence 
    from the second image to find the corresponding RGB values in the first. Since
    it is albedo this should ideally have 0 MSE loss
    """
    alb1 = alb1.to(device)
    alb2 = alb2.to(device)
    mask1 = mask1.to(device)
    alb1_predicted = warp_img(alb2, flow, device) * mask1
    imageio.imsave(
        f"loss_pics/alb_{i}_predicted.png",
        alb1_predicted.detach().cpu().permute(1, 2, 0).numpy(),
    )
    imageio.imsave(
        f"loss_pics/alb_{i}.png", alb1.detach().cpu().permute(1, 2, 0).numpy()
    )
    i += 1
    criterion = nn.MSELoss()
    loss = criterion(alb1_predicted, alb1 * mask1)
    return loss


def warp_img(im: torch.tensor, flow: np.array, device: torch.device) -> torch.tensor:
    """
    Warps image im according to the flow. n.b. im should be 1xCxHxW, flow should
    be 2xCxHxW
    """
    im = im.to(device)
    B, C, H, W = im.shape

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

    vgrid = vgrid.permute(0, 2, 3, 1).to(device)
    output = F.grid_sample(im, vgrid, align_corners=True)

    return output.to(device)
