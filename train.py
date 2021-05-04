import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dummy_model import DummyCNN
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.tiktok_dataset import TikTokDataset

parser = argparse.ArgumentParser(description="Train a model on the TikTok Dataset.")
optimizers = ["adam", "sgd"]
loss_fns = ["mse", "crossentropy"]

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
    "-batch", metavar="batch", help="Batch Size", type=int, default=32, required=False,
)

parser.add_argument(
    "-lr", metavar="lr", help="Learning Rate", type=float, default=1e-3, required=False,
)

parser.add_argument(
    "-optim",
    metavar="optimizer",
    choices=optimizers,
    help="Choose an optimizer",
    type=str,
    default="sgd",
    required=False,
)

parser.add_argument(
    "-loss",
    metavar="loss",
    choices=loss_fns,
    help="Choose a loss function",
    type=str,
    default="mse",
    required=False,
)

args = parser.parse_args()
ROOT_DIR = args.root_dir
N_EPOCHS = args.epochs
BATCH_SIZE = args.batch
LR = args.lr
OPTIM = args.optim
LOSS_FN = args.loss

if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)

    # Handle all data loading and related stuff
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_train = TikTokDataset(ROOT_DIR, device, transform=transform)
    dataset_test = TikTokDataset(ROOT_DIR, device, train=False, transform=transform)
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
    print("Data Loaded")

    # Handle all model related stuff
    model = DummyCNN()

    if OPTIM == optimizers[0]:
        optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    else:
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    if LOSS_FN == loss_fns[0]:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    print("Model and auxillary components initialized")

    # Train loop
    print("Beginning Training.")
    for epoch in range(N_EPOCHS):
        print(f"<Epoch {epoch}/{N_EPOCHS}>")
        running_loss = 0

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            mask = data["mask"]
            image = data["image"]

            optimizer.zero_grad()

            out = model(image)
            loss = criterion(out, mask.expand_as(out))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Loss: {running_loss / len(train_loader)}")

    print("Training Finished")
