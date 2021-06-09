import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from utils import squarize_image
from collections import defaultdict
from torch.utils.data.dataset import Dataset


class TikTokDataset(Dataset):
    def __init__(
        self,
        root_dir,
        device,
        train=True,
        transform=None,
        sample_size=4,
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

        for folder in os.listdir(appended_path)[:40]:
            if not os.path.isdir(appended_path + folder) or "234" in folder:
                continue

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
        self.image_paths = self.images[:]
        self.mask_paths = self.masks[:]

        TRAIN_TEST_SPLIT = round(0.8 * len(self.images))
        self.train_images = self.images[:TRAIN_TEST_SPLIT]
        self.train_masks = self.masks[:TRAIN_TEST_SPLIT]
        self.train_names = self.names[:TRAIN_TEST_SPLIT]
        self.train_image_paths = self.image_paths[:TRAIN_TEST_SPLIT]
        self.train_mask_paths = self.mask_paths[:TRAIN_TEST_SPLIT]

        self.test_images = self.images[TRAIN_TEST_SPLIT:]
        self.test_masks = self.masks[TRAIN_TEST_SPLIT:]
        self.test_names = self.names[TRAIN_TEST_SPLIT:]
        self.test_image_paths = self.image_paths[TRAIN_TEST_SPLIT:]
        self.test_mask_paths = self.mask_paths[TRAIN_TEST_SPLIT:]

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
            image_paths = self.train_image_paths
            mask_paths = self.train_mask_paths
        else:
            video_imgs = self.test_images
            video_masks = self.test_masks
            video_names = self.test_names
            image_paths = self.test_image_paths
            mask_paths = self.test_mask_paths

        images = np.array(
            list(
                map(
                    lambda im: squarize_image(
                        Image.open(im), self.squarize_size
                    ).numpy()
                    if self.squarize_size
                    else np.array(Image.open(im)),
                    video_imgs[idx],
                )
            )
        )
        masks = np.array(
            list(
                map(
                    lambda im: squarize_image(
                        Image.open(im), self.squarize_size
                    ).numpy()
                    if self.squarize_size
                    else np.array(Image.open(im)),
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
            "img_paths": image_paths[idx],
            "mask_paths": mask_paths[idx],
        }
