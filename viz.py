from matplotlib import image
import numpy as np
import torch
from torch.autograd import Variable
from models.factor_people.fact_people_ops import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import imageio


if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"

    device = torch.device(dev)
    all_dirs = get_model_dirs()
    fp = FactorsPeople(all_dirs, dev)

    img = "/phoenix/S3/ab2383/data/TikTok_dataset/00001/images/0004.png"
    img2 = "/phoenix/S3/ab2383/data/TikTok_dataset/00001/images/0005.png"
    mask = "/phoenix/S3/ab2383/data/TikTok_dataset/00001/masks/0004.png"
    mask2 = "/phoenix/S3/ab2383/data/TikTok_dataset/00001/masks/0005.png"

    img, mask = fp.get_image(img, mask)
    img2, mask2 = fp.get_image(img2, mask2)
    masked_img = img * mask
    imageio.imsave(
        "viz/img_original.png",
        img.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0,
    )
    imageio.imsave(
        "viz/img2.png", img2.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0,
    )

    imageio.imsave(
        "viz/masked_img.png",
        masked_img.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0,
    )

    flow = np.load("/phoenix/S3/ab2383/data/flows/4.npy")
    flow = flow[0]
    plt.subplot(121)
    plt.imshow(flow[0])
    plt.subplot(122)
    plt.imshow(flow[1])
    plt.savefig("viz/flow.png")

    B, C, H, W = img.shape
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
    output = F.grid_sample(torch.FloatTensor(img2), vgrid, align_corners=True)

    imageio.imsave(
        "viz/warped_img.png", output.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    )
