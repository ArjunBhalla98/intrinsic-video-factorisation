import os
import torch
import random
from PIL import Image
import numpy as np
from collections import defaultdict
from torch.utils.data.dataset import Dataset
import glob


class TikTokDataset(Dataset):
    def __init__(self, root_dir, device, train=True, transform=None, sample_size=1):
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
        self.images = defaultdict(list)
        self.masks = defaultdict(list)
        self.transform = transform
        self.train = train
        self.sample_size = sample_size
        appended_path = root_dir + "/TikTok_dataset"

        for folder in os.listdir(appended_path):
            video_id = int(folder) - 1

            self.images[video_id] = glob.glob(f"{appended_path}/{folder}/images/*.png")
            self.masks[video_id] = glob.glob(f"{appended_path}/{folder}/masks/*.png")

            assert len(self.images[video_id]) == len(
                self.masks[video_id]
            ), "Images Length != Masks Length"

        TRAIN_TEST_SPLIT = round(0.8 * len(self.images))
        self.train_images = {i: self.images[i] for i in range(TRAIN_TEST_SPLIT)}
        self.train_masks = {i: self.masks[i] for i in range(TRAIN_TEST_SPLIT)}
        self.test_images = {
            i: self.images[i] for i in range(TRAIN_TEST_SPLIT, len(self.images))
        }
        self.test_masks = {
            i: self.masks[i] for i in range(TRAIN_TEST_SPLIT, len(self.masks))
        }

    def __len__(self):
        if self.train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        print(idx)
        if self.train:
            video_imgs = self.train_images[idx]
            video_masks = self.train_masks[idx]
        else:
            video_imgs = self.test_images[idx]
            video_masks = self.test_masks[idx]

        start_idx = random.randint(0, len(video_imgs) - self.sample_size + 1)

        images = np.array(
            list(
                map(
                    lambda im: np.array(Image.open(im)),
                    video_imgs[start_idx : start_idx + self.sample_size],
                )
            )
        )
        masks = np.asarray(
            list(
                map(
                    lambda im: np.array(Image.open(im)),
                    video_masks[start_idx : start_idx + self.sample_size],
                )
            )
        )

        if self.transform:
            images = torch.cat(
                list(map(lambda im: torch.unsqueeze(self.transform(im), 0), images)),
                dim=0,
            )
            print("IMAGES SHAPE :" + str(images.size()))
            masks = torch.cat(
                list(map(lambda im: torch.unsqueeze(self.transform(im), 0), masks)),
                dim=0,
            )

        return {"images": images, "masks": masks}
