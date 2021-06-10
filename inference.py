import torch
import argparse
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from matplotlib.pyplot import imsave
import imageio
from data.tiktok_dataset import TikTokDataset
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

parser.add_argument("--save_dir", help="Path to save images", type=str, required=True)

parser.add_argument(
    "--dev", help="Cuda device if using GPU", type=str, default="0", required=False
)

args = parser.parse_args()
ROOT_DIR = args.root_dir
CUDA_DEV = args.dev
LOAD_PATH = args.load_state
SAVE_DIR = args.save_dir
BATCH_SIZE = 1

if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = f"cuda:{CUDA_DEV}"
    else:
        dev = "cpu"

    device = torch.device(dev)

    # Handle all data loading and related stuff
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_test = TikTokDataset(
        ROOT_DIR, device, train=False, transform=transform, sample_size=BATCH_SIZE,
    )
    test_loader = DataLoader(dataset_test, shuffle=True)
    print("Data Loaded")

    # Handle all model related stuff
    model = eval_model()
    model = model.to(device)
    model.eval()

    ##### PUT TASK SPECIFIC PRE-TRAINING THINGS HERE #####
    model_states_trained = {
        "self_shading_net": "models/states/fact_people_mse_ssn.pth",
        "shading_net": "models/states/fact_people_mse_sn.pth",
        "SH_model": "models/states/fact_people_mse_sh.pth",
        "albedo_net": "models/states/fact_people_mse_albedo.pth",
        "shadow_net": "models/states/fact_people_mse_shadow.pth",
        "refine_rendering_net": "models/states/fact_people_mse_rrn.pth",
    }
    all_dirs = get_model_dirs()
    factorspeople = FactorsPeople(all_dirs)
    factorspeople.load_model_state(model_states_trained)
    factorspeople.set_eval()
    # model.train_dropout = False  # relighting humans

    if LOAD_PATH:
        model.load_state_dict(torch.load(LOAD_PATH))

    print("Model and auxillary components initialized")

    print("Beginning Eval.")
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        masks = data["masks"].squeeze(0)
        images = data["images"].squeeze(0)
        name = data["names"].pop()[0]

        try:
            img, mask = factorspeople.get_image(
                data["img_paths"].pop()[0], data["mask_paths"].pop()[0]
            )
        except Exception:
            continue

        gt = (img.detach() * mask.detach()).squeeze().permute(1, 2, 0)
        out = (factorspeople.reconstruct(img, mask)[0]).squeeze().permute(1, 2, 0)

        print(gt.size(), out.size())
        imageio.imwrite(
            SAVE_DIR + "/" + name, out.detach().cpu().numpy().astype(np.uint8),
        )

        imageio.imwrite(
            SAVE_DIR + "/" + "gt_" + name, gt.detach().cpu().numpy().astype(np.uint8),
        )
        # imsave(name, rendering.detach().cpu().numpy())

    print(f"Eval Finished - images are in {SAVE_DIR}")
