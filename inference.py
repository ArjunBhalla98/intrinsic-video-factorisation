import os
import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from matplotlib.pyplot import imsave
import imageio
from tiktok_dataset import TikTokDataset
import torchvision
from models.factor_people.fact_people_ops import *

# To change loss or model, adjust these:
from models.relighting_model import CNNAE2ResNet as eval_model

parser = argparse.ArgumentParser(
    description="Generate recons images from a model on the TikTok Dataset."
)

parser.add_argument(
    "--root_dir",
    metavar="root",
    help="The root directory of your TikTok_dataset.",
    type=str,
    required=True,
)

parser.add_argument(
    "--load_state",
    metavar="load",
    help="Path to pretrained model weights, if req'd",
    type=str,
    required=False,
)

parser.add_argument("--save_dir", help="Path to save images", type=str, required=False)

parser.add_argument(
    "--dev", help="Cuda device if using GPU", type=str, default="0", required=False
)

parser.add_argument(
    "--log", help="Log if required on wandb", type=bool, default=False, required=False
)

parser.add_argument(
    "--load_prefix",
    help="Prefix for the models to load",
    type=str,
    required=False,
    default="opt_loss_reg_",
)

parser.add_argument("--batch_size", type=int)

args = parser.parse_args()
ROOT_DIR = args.root_dir
CUDA_DEV = args.dev
LOAD_PATH = args.load_state
SAVE_DIR = args.save_dir
LOG = args.log
BATCH_SIZE = args.batch_size
LOAD_PREFIX = args.load_prefix

if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = f"cuda:{CUDA_DEV}"
    else:
        dev = "cpu"

    device = torch.device(dev)
    device2 = torch.device("cuda:1")

    # Handle all data loading and related stuff
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_test = TikTokDataset(
        ROOT_DIR, device, train=True, transform=transform, sample_size=BATCH_SIZE,
    )
    test_loader = DataLoader(dataset_test, shuffle=False)
    print("Data Loaded")

    # Handle all model related stuff
    model = eval_model()
    model = model.to(device)
    model.eval()

    ##### PUT TASK SPECIFIC PRE-INFERENCE THINGS HERE #####
    model_states_trained = {
        "self_shading_net": f"models/states/{LOAD_PREFIX}ssn.pth",
        "shading_net": f"models/states/{LOAD_PREFIX}sn.pth",
        "SH_model": f"models/states/{LOAD_PREFIX}sh.pth",
        "albedo_net": f"models/states/{LOAD_PREFIX}albedo.pth",
        "shadow_net": f"models/states/{LOAD_PREFIX}shadow.pth",
        "refine_rendering_net": f"models/states/{LOAD_PREFIX}rrn.pth",
    }
    all_dirs = get_model_dirs()
    factorspeople = FactorsPeople(all_dirs, device=device)
    factorspeople.load_model_state(model_states_trained)
    factorspeople.set_eval()

    # nonft_factor_model = FactorsPeople(all_dirs, device=device)
    # nonft_factor_model.set_eval()
    # model.train_dropout = False  # relighting humans

    if LOAD_PATH:
        model.load_state_dict(torch.load(LOAD_PATH))

    if SAVE_DIR and not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    print("Model and auxillary components initialized")

    nonft_recons_error = 0
    ft_recons_error = 0
    mean_albedo_diff = 0
    mean_shading_diff = 0
    mean_recons_diff = 0
    count = 0
    recons_error_criterion = nn.MSELoss()

    print("Beginning Eval.")
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        torch.cuda.empty_cache()
        # masks = data["masks"].squeeze(0)
        # images = data["images"].squeeze(0)
        # name = data["names"].pop()[0]

        img2, mask2 = factorspeople.get_image(
            data["img_paths"][-1][0], data["mask_paths"][-1][0]
        )

        img, mask = factorspeople.get_image(
            data["img_paths"][-2][0], data["mask_paths"][-2][0]
        )

        img = img.to(device)
        mask = mask.to(device)

        img2 = img2.to(device2)
        mask2 = mask2.to(device2)
        # images = images.to(device)
        # masks = masks.to(device)
        factorspeople.to(device)

        gt = img.detach() * mask.detach() * 255.0
        gt_img = gt.squeeze().permute(1, 2, 0)

        # nonft_reconstruction, nonft_factors = nonft_factor_model.reconstruct(img, mask)
        # nonft_out = (
        #     nonft_reconstruction.detach()
        #     * mask.detach()
        #     * 255.0
        #     # .squeeze()
        #     # .permute(1, 2, 0)
        # )
        # nonft_recons_error += recons_error_criterion(
        #     nonft_out.squeeze().permute(1, 2, 0), gt_img
        # ).item()

        torch.cuda.empty_cache()

        reconstruction, factors = factorspeople.reconstruct(img, mask)
        out = (
            reconstruction.detach()
            * mask.detach()
            * 255.0
            # .squeeze().permute(1, 2, 0)
        )

        factorspeople.to(device2)

        reconstruction2, factors2 = factorspeople.reconstruct(img2, mask2)
        out2 = reconstruction2.detach() * mask2.detach() * 255.0
        ft_recons_error += recons_error_criterion(
            out.squeeze().permute(1, 2, 0), gt_img
        ).item()

        albedo = factors["albedo"].to(device)
        albedo2 = factors2["albedo"].to(device)

        shading = factors["shading"].to(device)
        shading2 = factors2["shading"].to(device)

        mean_albedo_diff += recons_error_criterion(albedo, albedo2)
        mean_shading_diff += recons_error_criterion(shading, shading2)
        mean_recons_diff += recons_error_criterion(reconstruction, reconstruction2)

        count += 1

        if SAVE_DIR:
            # out_np = out.detach().cpu().numpy()
            # nonft_out_np = nonft_out.detach().cpu().numpy()
            # gt_np = gt.detach().cpu().numpy()
            # shading = factors["shading"].squeeze(0).permute(1, 2, 0)
            shading = factors["shading"].detach()
            shading = shading / shading.max() * 255.0
            # albedo = factors["albedo"].squeeze(0).permute(1, 2, 0)
            albedo = factors["albedo"].detach()
            albedo = albedo / albedo.max() * 255.0

            # nonft_albedo = nonft_factors["albedo"].detach()
            # nonft_albedo = nonft_albedo / nonft_albedo.max() * 255.0

            # nonft_shading = nonft_factors["shading"].detach()
            # nonft_shading = nonft_shading / nonft_shading.max() * 255.0

            batch_out_ft = torch.cat((out.detach(), albedo, shading, gt.detach()))
            # batch_out_nonft = torch.cat(
            #     (nonft_out.detach(), nonft_albedo, nonft_shading, gt.detach())
            # )
            out_ft = torchvision.utils.make_grid(batch_out_ft, nrow=4)
            # out_nonft = torchvision.utils.make_grid(batch_out_nonft, nrow=4)

            imageio.imwrite(f"{SAVE_DIR}/{name}", out_ft.permute(1, 2, 0).cpu().numpy())
            # imageio.imwrite(
            #     f"{SAVE_DIR}/nonft_{name}", out_nonft.permute(1, 2, 0).cpu().numpy()
            # )

            # shading_np = shading.detach().cpu().numpy()
            # albedo_np = albedo.detach().cpu().numpy()
            # name_noext = name[: name.find(".")]
            # np.save(SAVE_DIR + "/" + name_noext + ".npy", out_np)
            # np.save(SAVE_DIR + "/gt_" + name_noext + ".npy", gt_np)
            # np.save(SAVE_DIR + "/shading_" + name_noext + ".npy", shading_np)
            # np.save(SAVE_DIR + "/albedo_" + name_noext + ".npy", albedo_np)

            # imageio.imwrite(
            #     SAVE_DIR + "/" + name, out_np.astype(np.uint8),
            # )

            # imageio.imwrite(
            #     SAVE_DIR + "/" + "nonft_" + name, nonft_out_np.astype(np.uint8),
            # )

            # imageio.imwrite(
            #     SAVE_DIR + "/" + "shading_" + name, shading_np.astype(np.uint8),
            # )

            # imageio.imwrite(
            #     SAVE_DIR + "/" + "albedo_" + name, albedo_np.astype(np.uint8),
            # )
            # # imsave(name, rendering.detach().cpu().numpy())

    if SAVE_DIR:
        print(f"Eval Finished - images are in {SAVE_DIR}")
    else:
        print("Eval Finished")

    print(f"Average reconstruction consistency: {mean_recons_diff / count}")
    print(f"Average Albedo consistency: {mean_albedo_diff / count}")
    print(f"Average Shading consistency: {mean_shading_diff / count}")
    print(
        f"Average Validation Set Reconstruction Error, Fine Tuned: {ft_recons_error / count}"
    )
    # print(
    #     f"Average Validation Set Reconstruction Error, NON Fine Tuned: {nonft_recons_error / count}"
    # )
