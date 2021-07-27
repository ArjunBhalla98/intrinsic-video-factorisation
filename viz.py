import numpy as np
import torch
from models.factor_people.fact_people_ops import *
import matplotlib.pyplot as plt
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
    imageio.imsave(
        "viz/get_image_viz.png",
        img.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0,
    )

    flow = np.load("/phoenix/S3/ab2383/data/flows/1.npy")
    flow = flow[0]
    plt.subplot(121)
    plt.imshow(flow[0])
    plt.subplot(122)
    plt.imshow(flow[1])
    plt.savefig("viz/flow.png")
#     imageio.imsave("viz/flow_x.png", flow[0])
#     imageio.imsave("viz/flow_y.png", flow[1])
