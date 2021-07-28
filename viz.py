from matplotlib import image
import numpy as np
import torch
from torch.autograd import Variable
from models.factor_people.fact_people_ops import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import imageio
from loss.unsupervised_loss import warp_img


if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"

    device = torch.device(dev)
    all_dirs = get_model_dirs()
    fp = FactorsPeople(all_dirs, dev)

    img_path = "/phoenix/S3/ab2383/data/TikTok_dataset/00001/images/0001.png"
    img2_path = "/phoenix/S3/ab2383/data/TikTok_dataset/00001/images/0002.png"
    mask_path = "/phoenix/S3/ab2383/data/TikTok_dataset/00001/masks/0001.png"
    mask2_path = "/phoenix/S3/ab2383/data/TikTok_dataset/00001/masks/0002.png"

    img, mask = fp.get_image(img_path, mask_path)
    img2, mask2 = fp.get_image(img2_path, mask2_path)
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

    flow = np.load("/phoenix/S3/ab2383/data/flows/1.npy")
    flow = flow[0]
    imageio.imsave("flowx.png", flow[0])
    imageio.imsave("flowy.png", flow[1])
    flowx = fp.get_image("flowx.png", mask_path).squeeze(0)
    flowy = fp.get_image("flowy.png", mask_path).squeeze(0)
    flow = torch.cat((flowx, flowy), 0)
    plt.subplot(121)
    plt.imshow(flow[0])
    plt.subplot(122)
    plt.imshow(flow[1])
    plt.savefig("viz/flow.png")

    output = warp_img(img2, np.expand_dims(flow, 0), dev)

    imageio.imsave(
        "viz/warped_img.png", output.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    )
