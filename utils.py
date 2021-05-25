import torch
import sys
from PIL import ImageOps, Image
import torchvision.transforms as transforms


def squarize_image(image: Image.Image, new_size: int = 1024) -> torch.Tensor:
    """
    Takes images as PIL files (i.e. pass the results of Image.open() to this) and pads
    them to the given size, then returns a 0-1 normalized tensor. Make sure that the Image being 
    passed in has 0-255 range values.
    """

    w, h = image.size
    if w != h:
        side_padding = (new_size - w) // 2
        tops_padding = (new_size - h) // 2
        padding = (side_padding, tops_padding, side_padding, tops_padding)
        img = ImageOps.expand(image, padding)

        transform = transforms.Compose(
            [transforms.CenterCrop([new_size, new_size]), transforms.ToTensor()]
        )
        img = transform(img)

        img = img / 255.0

    return img.permute(1, 2, 0)
