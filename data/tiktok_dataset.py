import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset
import glob
import torchvision


class TikTokDataset(Dataset):
    def __init__(self, root_dir, device, train=True, transform=None):
        self.images = []
        self.masks = []
        self.transform = transform
        self.train = train
        appended_path = root_dir + "/TikTok_dataset"

        # This should be ok because Image.open lazy loads, keeping memory clear
        for folder in os.listdir(appended_path):
            self.images += glob.glob(f"{appended_path}/{folder}/images/*.png")
            self.masks += glob.glob(f"{appended_path}/{folder}/masks/*.png")

        assert len(self.images) == len(self.masks), "Images Length != Masks Length"
        TRAIN_TEST_SPLIT = int(0.8 * len(self.images))

        self.train_images = self.images[:TRAIN_TEST_SPLIT]
        self.train_masks = self.masks[:TRAIN_TEST_SPLIT]
        self.test_images = self.images[TRAIN_TEST_SPLIT:]
        self.test_masks = self.masks[TRAIN_TEST_SPLIT:]

    def __len__(self):
        if self.train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            image = Image.open(self.train_images[idx])
            mask = Image.open(self.train_masks[idx])
        else:
            image = Image.open(self.test_images[idx])
            mask = Image.open(self.test_masks[idx])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {"image": image, "mask": mask}
