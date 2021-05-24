import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.tiktok_dataset import TikTokDataset

# To change loss or model, adjust these:
from loss.unsupervised_loss import l1_reconstruction_loss as criterion
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

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # for running relighting humans
            images = 2.0 * images - 1
            gt = (images * masks).to(device)

            transport, albedo, light = model(gt)
            transport = transport.view(BATCH_SIZE, 1024 * 1024, 9).to(device)
            albedo = albedo.permute(0, 2, 3, 1).to(device)
            shading = (
                (transport @ light).view(BATCH_SIZE, 1024, 1024, 3).to(device)
            )  # get rid of the magic numbers at some point if we use this properly
            rendering = (albedo * shading).permute(0, 3, 1, 2).to(device)

            loss = criterion(rendering, gt)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Loss: {running_loss / len(train_loader)}")

    print("Training Finished")

    if SAVE_PATH:
        torch.save(training_model.state_dict(), SAVE_PATH)
