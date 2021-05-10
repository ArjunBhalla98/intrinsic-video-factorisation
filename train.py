import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.tiktok_dataset import TikTokDataset

# To change loss or model, adjust these:
from loss.unsupervised_loss import l1_loss as criterion
from models.dummy_model import DummyCNN as training_model

parser = argparse.ArgumentParser(description="Train a model on the TikTok Dataset.")

parser.add_argument(
    "--root_dir",
    metavar="root",
    help="The root directory of your TikTok_dataset. Do not include the dataset path, just the folder containing it.",
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
    "--batch", metavar="batch", help="Batch Size", type=int, default=32, required=False,
)

parser.add_argument(
    "-lr", metavar="lr", help="Learning Rate", type=float, default=1e-3, required=False,
)

args = parser.parse_args()
ROOT_DIR = args.root_dir
N_EPOCHS = args.epochs
BATCH_SIZE = args.batch
LR = args.lr

if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)

    # Handle all data loading and related stuff
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_train = TikTokDataset(
        ROOT_DIR, device, transform=transform, sample_size=BATCH_SIZE
    )
    dataset_test = TikTokDataset(
        ROOT_DIR, device, train=False, transform=transform, sample_size=BATCH_SIZE
    )
    train_loader = DataLoader(dataset_train, shuffle=True)
    test_loader = DataLoader(dataset_test, shuffle=True)
    print("Data Loaded")

    # Handle all model related stuff
    model = training_model()

    # Do the loss function / model thing for this too if needed
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))

    print("Model and auxillary components initialized")

    # Train loop
    print("Beginning Training.")
    for epoch in range(N_EPOCHS):
        print(f"<Epoch {epoch}/{N_EPOCHS}>")
        running_loss = 0

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            masks = data["masks"].squeeze(0)
            images = data["images"].squeeze(0)

            optimizer.zero_grad()

            out = model(images)
            loss = criterion(out)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Loss: {running_loss / len(train_loader)}")

    print("Training Finished")
