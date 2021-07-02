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


class TikTokDataset(Dataset):
    def __init__(
        self,
        root_dir,
        device,
        train=True,
        transform=None,
        sample_size=1,
        squarize_size=None,
    ):
        """
        Initialises the TikTok Dataset. This will return, upon "getitem", a sample that is
        temporally contiguous from a random video, of length sample_size. This function is built to
        "lazy load" so as not to clog up memory.

        root_dir : string = Path to base of TikTok dataset directory. Does not include "TikTok_dataset".
        device : torch.device = Current device being used (GPU or CPU pretty much)
        train : bool = Do you want the train or test set?
        transform : torchvision.transforms = Transforms applied to each image on return
        sample_size : int = Number of images per sample drawn from this dataset.
        """
        self.images = []
        self.masks = []
        self.names = []
        self.image_paths = []
        self.mask_paths = []
        self.transform = transform
        self.train = train
        self.sample_size = sample_size
        self.squarize_size = squarize_size
        appended_path = root_dir
        self.smaller_width = 278
        self.smaller_height = 500

        video_dirs = os.listdir(appended_path)[:40]
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

            video_imgs = glob.glob(f"{appended_path}/{folder}/images/*.png")
            video_masks = glob.glob(f"{appended_path}/{folder}/masks/*.png")

            self.images += [
                video_imgs[i : i + sample_size]
                for i in range(0, len(video_imgs), sample_size)
            ]
            self.masks += [
                video_masks[i : i + sample_size]
                for i in range(0, len(video_masks), sample_size)
            ]

            self.names += list(
                map(
                    lambda nested_paths: list(
                        map(lambda x: f"{folder}_{x[x.rfind('/')+1:]}", nested_paths)
                    ),
                    (
                        video_imgs[i : i + sample_size]
                        for i in range(0, len(video_imgs), sample_size)
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

        images = np.array(
            list(
                map(
                    lambda im: squarize_image(
                        Image.open(im).resize(
                            (self.smaller_width, self.smaller_height)
                        ),
                        self.squarize_size,
                    ).numpy()
                    if self.squarize_size
                    else np.array(
                        Image.open(im).resize(
                            (self.smaller_width, self.smaller_height),
                            resample=Image.BICUBIC,
                        )
                    ),
                    video_imgs[idx],
                )
            )
        )
        masks = np.array(
            list(
                map(
                    lambda im: squarize_image(
                        Image.open(im).resize(
                            (self.smaller_width, self.smaller_height)
                        ),
                        self.squarize_size,
                    ).numpy()
                    if self.squarize_size
                    else np.array(
                        Image.open(im).resize(
                            (self.smaller_width, self.smaller_height),
                            resample=Image.BICUBIC,
                        )
                    ),
                    video_masks[idx],
                )
            )
        )

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
    # Test script
    root_dir = "/Users/arjunbhalla/Desktop/TikTok_dataset/"
    device = torch.device("cpu")
    dataset = TikTokDataset(root_dir, device)
    dataloader = DataLoader(dataset)

    for _, data in enumerate(dataloader):
        masks = data["masks"].squeeze(0)
        images = data["images"].squeeze(0)
        image = images.squeeze(0).detach().numpy()
        mask = masks.squeeze(0).unsqueeze(-1).detach().numpy() / 255.0

        imageio.imsave("./dataloader_test_raw.png", image)
        imageio.imsave("./dataloader_test_mask.png", mask)
        imageio.imsave("./dataloader_test_combined.png", image * mask)
        break

