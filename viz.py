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
    old_flow_min = np.min(flow)
    flow_old = np.copy(flow)
    flow -= old_flow_min
    old_flow_max = np.max(flow)
    flow /= old_flow_max
    print(flow.min(), flow.max())
    imageio.imsave("flowx.png", np.expand_dims(flow[0], 2))
    imageio.imsave("flowy.png", np.expand_dims(flow[1], 2))
    flowx, _ = fp.get_image("flowx.png", mask_path)
    flowy, _ = fp.get_image("flowy.png", mask_path)
    flowx = flowx.squeeze(0).mean(0, keepdim=True)
    flowy = flowy.squeeze(0).mean(0, keepdim=True)
    flow = torch.cat((flowx, flowy), 0)
    flow *= old_flow_max
    flow += old_flow_min
    print(
        np.min(flow_old),
        np.max(flow_old),
        flow.min(),
        flow.max(),
        old_flow_min,
        old_flow_max,
    )
    plt.subplot(121)
    plt.imshow(flow[0])
    plt.subplot(122)
    plt.imshow(flow[1])
    plt.savefig("viz/flow.png")

    print(flow.size(), img2.size())
    output = warp_img(img2, np.expand_dims(flow, 0), dev)
    print(
        ((output.cpu() - img2.cpu()) ** 2).mean(),
        ((output.cpu() - img.cpu()) ** 2).mean(),
    )

    imageio.imsave(
        "viz/warped_img.png", output.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    )
