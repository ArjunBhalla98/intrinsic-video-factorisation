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
from data.tiktok_dataset import TikTokDataset
import imageio

# To change loss or model, adjust these:
from loss.unsupervised_loss import l1_reconstruction_loss as criterion

criterion = nn.MSELoss()
from models.relighting_model import CNNAE2ResNet as training_model

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

args = parser.parse_args()
ROOT_DIR = args.root_dir
N_EPOCHS = args.epochs
CUDA_DEV = args.dev
BATCH_SIZE = args.batch
LR = args.lr
LOAD_PATH = args.load_state
SAVE_PATH = args.save_state

if __name__ == "__main__":
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
        squarize_size=1024,
    )
    dataset_test = TikTokDataset(
        ROOT_DIR,
        device,
        train=False,
        transform=transform,
        sample_size=BATCH_SIZE,
        squarize_size=1024,
    )
    train_loader = DataLoader(dataset_train, shuffle=True)
    test_loader = DataLoader(dataset_test, shuffle=True)
    print("Data Loaded")

    # Handle all model related stuff
    model = training_model()
    model = model.to(device)

    if LOAD_PATH:
        model.load_state_dict(torch.load(LOAD_PATH))

    # Do the loss function / model thing for this too if needed
    # optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0)

    print("Model and auxillary components initialized")

    # Train loop
    print("Beginning Training.")
    for epoch in range(N_EPOCHS):
        print(f"<Epoch {epoch}/{N_EPOCHS}>")
        running_loss = 0

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            masks = data["masks"].squeeze(0)
            images = data["images"].squeeze(0)

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            mask3 = (
                Variable(
                    torch.from_numpy(
                        np.repeat(masks.detach().cpu().numpy(), 3, axis=0)
                    ),
                    requires_grad=True,
                )
                .squeeze(1)
                .to(device)
            )
            mask9 = (
                Variable(
                    torch.from_numpy(
                        np.repeat(masks.detach().cpu().numpy(), 9, axis=0)
                    ),
                    requires_grad=True,
                )
                .squeeze(1)
                .to(device)
            )

            # for running relighting humans
            images = 2.0 * images - 1
            gt = (images * mask3).to(device)

            transport, albedo, light = model(gt)
            transport = Variable((mask9 * transport).data[0], requires_grad=True)
            albedo = Variable((albedo * mask3).data[0], requires_grad=True)
            light = Variable(light.data, requires_grad=True)
            transport = transport.permute(1, 2, 0).to(device)
            albedo = albedo.permute(1, 2, 0).to(device)
            shading = torch.matmul(transport, light).to(device)
            rendering = (albedo * shading * 255.0).to(device)

            print((rendering.permute(2, 0, 1) - gt.squeeze(0)))
            loss = criterion(rendering.permute(2, 0, 1), gt.squeeze(0))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_batch_loss = running_loss / len(train_loader)
        wandb.log({"loss": epoch_batch_loss})
        print(f"Loss: {epoch_batch_loss}")

    print("Training Finished")

    if SAVE_PATH:
        torch.save(model.state_dict(), SAVE_PATH)
