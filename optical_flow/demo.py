import sys

sys.path.append("core")

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
import pickle

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from fact_people_ops import FactorsPeople


if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DEVICE = torch.device(DEVICE)
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def load_image(imfile):
    img = np.array(Image.open(imfile).resize((278, 500))).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)
    # return img


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow("image", img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for i, folder in enumerate(sorted(os.listdir(args.path))[:100]):
            print(f"video {i+1}")
            images = glob.glob(
                os.path.join(args.path + folder + "/images/", "*.png")
            ) + glob.glob(os.path.join(args.path + folder + "/images/", "*.jpg"))

            masks = glob.glob(
                os.path.join(args.path + folder + "/masks/", "*.png")
            ) + glob.glob(os.path.join(args.path + folder + "/masks/", "*.jpg"))

            fact_people = FactorsPeople(DEVICE)

            ret = []
            is_ret = False
            images = sorted(images)[:100]
            masks = sorted(masks)[:100]
            count = 0
            for i, imfiles in enumerate(zip(images[:-1], images[1:])):
                if count >= 100:
                    print("count >= 100")
                    break
                imfile1 = imfiles[0]
                imfile2 = imfiles[1]

                image1, _ = fact_people.get_image(imfile1, masks[i])
                image2, _ = fact_people.get_image(imfile2, masks[i + 1])

                image1 = image1.to(DEVICE)
                image2 = image2.to(DEVICE)

                image1 *= 255.0
                image2 *= 255.0

                _, flow_up = model(image1, image2, iters=20, test_mode=True)
                if not is_ret:
                    is_ret = True
                    ret = flow_up
                else:
                    ret = torch.cat((ret, flow_up), 0)
                count += 1

            np.save(
                f"/phoenix/S3/ab2383/data/flows/{i+1}.npy", ret.detach().cpu().numpy()
            )
            if i == 99:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--path", help="dataset for evaluation")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )
    args = parser.parse_args()

    demo(args)
