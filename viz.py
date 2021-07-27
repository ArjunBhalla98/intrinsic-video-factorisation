import numpy as np
import torch
from .models.factor_people.fact_people_ops import *
import imageio


if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"

    device = torch.device(dev)
    all_dirs = get_model_dirs()
    fp = FactorsPeople(all_dirs, dev)

    img = "/phoenix/S3/ab2383/data/TikTok_dataset/00001/images/0001.png"
    mask = "/phoenix/S3/ab2383/data/TikTok_dataset/00001/masks/0001.png"

    img, mask = fp.get_image(img, mask)
    imageio.imsave("get_image_viz.png", img.detach().permute(1, 2, 0).cpu().numpy())

    flow = np.load("/phoenix/S3/ab2383/data/flows/1.npy")
    flow = flow[0]
    imageio.imsave("flow_x.png", flow[0])
    imageio.imsave("flow_y.png", flow[1])
