from PIL import Image
import torch
import wandb
import torch.nn as nn
from torch.autograd import Variable
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tiktok_dataset import TikTokDataset
from models.RAFT.raft import RAFT
from models.RAFT.utils import InputPadder
from models.factor_people.fact_people_ops import *
from loss.unsupervised_loss import optical_flow_loss

# To change loss or model, adjust these:
from loss.unsupervised_loss import l2_mse_loss as criterion

# from models.relighting_model import CNNAE2ResNet as training_model

parser = argparse.ArgumentParser(description="Train a model on the TikTok Dataset.")

parser.add_argument(
    "--root_dir",
    metavar="root",
    help="The root directory of your TikTok_dataset.",
    type=str,
    required=True,
)

parser.add_argument(
    "--epochs",
    metavar="epochs",
    help="Number of epochs to train for",
    type=int,
    default=25,
    required=False,
)

parser.add_argument(
    "--load_state",
    metavar="load",
    help="Path to pretrained model weights, if req'd",
    type=str,
    required=False,
)

parser.add_argument(
    "--save_state",
    metavar="save",
    help="Path to save trained model weights, if req'd",
    type=str,
    required=False,
)

parser.add_argument(
    "--batch", metavar="batch", help="Batch Size", type=int, default=32, required=False,
)

parser.add_argument(
    "-lr", metavar="lr", help="Learning Rate", type=float, default=1e-3, required=False,
)

parser.add_argument(
    "--dev", help="Cuda device if using GPU", type=str, default="0", required=False
)

##### RAFT arguments to satisfy the args argument for init

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

#######

parser.add_argument(
    "--raft_path",
    help="Path to the RAFT .pth",
    default="models/states/raft-things.pth",
    type=str,
    required=False,
)


args = parser.parse_args()
ROOT_DIR = args.root_dir
N_EPOCHS = args.epochs
CUDA_DEV = args.dev
BATCH_SIZE = args.batch
LR = args.lr
LOAD_PATH = args.load_state
SAVE_PATH = args.save_state
RAFT_PATH = args.raft_path

if __name__ == "__main__":
    torch.cuda.empty_cache()
    wandb.login()
    wandb.init(project="video-factorisation", entity="arjunb")
    config = wandb.config

    if torch.cuda.is_available():
        dev = f"cuda:{CUDA_DEV}"
    else:
        dev = "cpu"

    device = torch.device(dev)

    # Handle all data loading and related stuff
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_train = TikTokDataset(
        ROOT_DIR,
        device,
        transform=transform,
        sample_size=BATCH_SIZE,
        # squarize_size=1024,
    )
    train_loader = DataLoader(dataset_train, shuffle=True)
    print("Data Loaded")

    # Handle all model related stuff
    # model = training_model()
    # model = model.to(device)

    # if LOAD_PATH:
    #     model.load_state_dict(torch.load(LOAD_PATH))

    ##### PUT TASK SPECIFIC PRE-TRAINING THINGS HERE #####
    all_dirs = get_model_dirs()
    factorspeople = FactorsPeople(all_dirs, device)
    raft = RAFT(args)
    raft = torch.nn.DataParallel(RAFT(args))
    if RAFT_PATH:
        raft.load_state_dict(torch.load(RAFT_PATH, map_location=device))
        raft_dev = torch.device("cpu")

    raft = raft.module
    raft.to(raft_dev)
    raft.eval()
    optical_lambda = 0.1

    static_factor_model = FactorsPeople(all_dirs, device)
    static_factor_model.set_eval()
    shading_albedo_loss = nn.MSELoss()
    shading_lambda = 0.1
    albedo_lambda = 0.1
    ######################################################

    # Do the loss function / model thing for this too if needed
    params = (
        list(factorspeople.self_shading_net.parameters())
        + list(factorspeople.shading_net.parameters())
        + list(factorspeople.SH_model.parameters())
        + list(factorspeople.albedo_net.parameters())
        + list(factorspeople.shadow_net.parameters())
        + list(factorspeople.refine_rendering_net.parameters())
    )
    # params = model.parameters()
    optimizer = optim.Adam(params, lr=LR, betas=(0.9, 0.999))

    print("Model and auxillary components initialized")

    # Train loop
    print("Beginning Training.")
    for epoch in range(N_EPOCHS):
        print(f"<Epoch {epoch}/{N_EPOCHS}>")
        running_loss = 0

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # masks = data["masks"].squeeze(0)
            # images = data["images"].squeeze(0)

            # images = images.to(device)
            # masks = masks.to(device)

            optimizer.zero_grad()
            #### PUT MODEL SPECIFIC FORWARD PASS CODE HERE ####
            #### FOR SIGGRAPH TRAINING ####
            img, mask = factorspeople.get_image(
                data["img_paths"].pop()[0], data["mask_paths"].pop()[0]
            )

            img2, mask2 = factorspeople.get_image(
                data["img_paths"].pop()[0], data["mask_paths"].pop()[0]
            )

            padder = InputPadder(img.shape)
            img, img2 = padder.pad(img, img2)
            mask, mask2 = padder.pad(mask, mask2)

            img = img.to(device)
            mask = mask.to(device)

            img2 = img2.to(device)
            mask2 = mask2.to(device)

            gt = img.detach() * mask.detach()
            out, factors = factorspeople.reconstruct(img, mask)
            _, static_factors = static_factor_model.reconstruct(img, mask)
            _, static_factors_2 = static_factor_model.reconstruct(img2, mask2)
            static_shading = static_factors["shading"]
            static_shading = static_shading / static_shading.max() * 255.0
            static_albedo = static_factors["albedo"]
            static_albedo = static_albedo / static_albedo.max() * 255.0
            static_albedo_2 = static_factors_2["albedo"]
            static_albedo_2 = static_albedo_2 / static_albedo_2.max() * 255.0

            shading = factors["shading"]
            shading = shading / shading.max() * 255.0

            albedo = factors["albedo"]
            albedo = albedo / albedo.max() * 255.0

            _, flow = raft(img, img2, iters=20, test_mode=True)

            optical_loss = (
                optical_flow_loss(albedo, static_albedo_2, mask, flow) * optical_lambda
            )
            shading_loss = shading_albedo_loss(static_shading, shading) * shading_lambda
            albedo_loss = shading_albedo_loss(static_albedo, albedo) * albedo_lambda
            ####################################################
            # add shading loss and albedo loss to this for the SIGGRAPH
            reconstruction_loss = criterion(out, gt)
            loss = reconstruction_loss + optical_loss + shading_loss + albedo_loss
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            wandb.log(
                {
                    "loss": loss.item(),
                    "shading loss": shading_loss.item(),
                    "albedo loss": albedo_loss.item(),
                    "optical loss": optical_loss.item(),
                }
            )

        epoch_batch_loss = running_loss / len(train_loader)
        print(f"Loss: {epoch_batch_loss}")

    print("Training Finished")

    if SAVE_PATH:
        #     torch.save(model.state_dict(), SAVE_PATH + ".pth")

        # SIGGRAPH Model
        torch.save(factorspeople.self_shading_net.state_dict(), SAVE_PATH + "ssn.pth")
        torch.save(factorspeople.shading_net.state_dict(), SAVE_PATH + "sn.pth")
        torch.save(factorspeople.SH_model.state_dict(), SAVE_PATH + "sh.pth")
        torch.save(factorspeople.albedo_net.state_dict(), SAVE_PATH + "albedo.pth")
        torch.save(factorspeople.shadow_net.state_dict(), SAVE_PATH + "shadow.pth")
        torch.save(
            factorspeople.refine_rendering_net.state_dict(), SAVE_PATH + "rrn.pth"
        )

