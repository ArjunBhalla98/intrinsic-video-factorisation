import os
import glob
import torch
import random
import imageio
import numpy as np
from PIL import Image
from utils import squarize_image
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class YoutubeDataset(Dataset):
  def __init__(
    self,
    root_dir,
    device,
    train=True,
    transform=None
  ):
    self.images = []
    self.masks = []
    self.names = []
    self.image_paths = []
    self.mask_paths = []
    self.train = train
    self.transform=transform
    appended_path = root_dir

    video_dirs = os.listdir(appended_path) #what is this? why [:40]?
    TRAIN_TEST_SPLIT = round(0.8 * len(video_dirs))

    for i, folder in enumerate(video_dirs):
        if not os.path.isdir(appended_path + folder):
            continue 

        if i == TRAIN_TEST_SPLIT:
            self.train_images = self.images[:]
            self.train_masks = self.masks[:]
            self.train_names = self.names[:]

            self.images = []
            self.masks = []
            self.names = []

        video_imgs = glob.glob(f"{appended_path}/{folder}/img/*.jpg")
        video_masks = glob.glob(f"{appended_path}/{folder}/mask/*.png")

        self.images += [
            video_imgs[i : i + 1]
            for i in range(0, len(video_imgs))
        ]
        self.masks += [
            video_masks[i : i + 1]
            for i in range(0, len(video_masks))
        ]

        self.names += list(
            map(
                lambda nested_paths: list(
                    map(lambda x: f"{folder}_{x[x.rfind('/')+1:]}", nested_paths)
                ),
                (
                    video_imgs[i : i + 1]
                    for i in range(0, len(video_imgs))
                ),
            )
        )

        assert len(self.images) == len(self.masks), "Images Length != Masks Length"

    self.test_images = self.images[:]
    self.test_masks = self.masks[:]
    self.test_names = self.names[:]





  def __len__(self):
    if self.train:
      return len(self.train_images)
    else:
      return len(self.test_images)

  
  def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      if self.train:
          video_imgs = self.train_images
          video_masks = self.train_masks
          video_names = self.train_names
      else:
          video_imgs = self.test_images
          video_masks = self.test_masks
          video_names = self.test_names
    
      images = video_imgs[idx]

      masks = video_masks[idx]

      if self.transform:
          images = torch.cat(
              list(map(lambda im: torch.unsqueeze(self.transform(im), 0), images)),
              dim=0,
          )

          masks = torch.cat(
              list(map(lambda im: torch.unsqueeze(self.transform(im), 0), masks)),
              dim=0,
          )

      return {
          "images": images,
          "masks": masks,
          "names": video_names[idx],
          "img_paths": video_imgs[idx],
          "mask_paths": video_masks[idx],
      }



if __name__ == "__main__":
    root_dir = "/phoenix/S3/alh293/data/youtube/frames"

    if torch.cuda.is_available():
       device = torch.device("cuda")
    else:
     device = torch.device("cpu")

    dataset = YoutubeDataset(root_dir, device)
    dataloader = DataLoader(dataset)

    # for _, data in enumerate(dataloader):
    #     masks = data["masks"].squeeze(0)
    #     images = data["images"].squeeze(0)
    #     image = images.squeeze(0).detach().numpy()
    #     mask = masks.squeeze(0).unsqueeze(-1).detach().numpy() / 255.0

    #     imageio.imsave("./dataloader_test_raw.png", image)
    #     imageio.imsave("./dataloader_test_mask.png", mask)
    #     imageio.imsave("./dataloader_test_combined.png", image * mask)
    #     break
