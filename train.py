import torch
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.tiktok_dataset import TikTokDataset
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Train a model on the TikTok Dataset.")
parser.add_argument(
    "-root_dir",
    metavar="root",
    nargs=1,
    help="The root directory of your TikTok_dataset. Do not include the dataset path, just the folder containing it.",
    type=str,
)

args = parser.parse_args()
if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    transform = transforms.Compose([transforms.ToTensor()])
    device = torch.device(dev)
    tmp_base = "/Users/arjunbhalla/Desktop"
    dataset = TikTokDataset(tmp_base, device, transform)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    for i, data in enumerate(train_loader):
        sample = data
        mask = sample["mask"]
        image = sample["image"].detach().numpy().transpose(1, 2, 0)

    #     plt.imshow(image)
    #     plt.show()
    #     break
