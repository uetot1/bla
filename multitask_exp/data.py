import random

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from train_base import VimeoSeptuplet


class VimeoSeptupletFlip(VimeoSeptuplet):
    """Vimeo-90K septuplet clips: random crop + clip-level horizontal flip.

    Reuses train_base.VimeoSeptuplet only for list/folder validation. Loading
    and augmentation are implemented here so the result does not depend on
    which version of train_base.py is checked out (the committed one crops
    without flipping; a local experimental edit also flips).
    The paper describes random crops with horizontal flipping.
    """

    def __init__(self, *args, hflip=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.hflip = hflip

    def __getitem__(self, index):
        frame_count = self.group_size + 1
        start = random.randint(1, 8 - frame_count) if self.random_crop else 1
        folder = self.sequence_root / self.sequences[index]
        frames = []
        for frame_index in range(start, start + frame_count):
            with Image.open(folder / f"im{frame_index}.png") as image:
                frames.append(to_tensor(image.convert("RGB")))
        height, width = frames[0].shape[-2:]
        if any(frame.shape[-2:] != (height, width) for frame in frames):
            raise ValueError(f"Frame sizes differ in {folder}")
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(f"{folder} is smaller than {self.crop_size}x{self.crop_size}")
        if self.random_crop:
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
        else:
            top = (height - self.crop_size) // 2
            left = (width - self.crop_size) // 2
        clip = torch.stack([
            frame[:, top:top + self.crop_size, left:left + self.crop_size]
            for frame in frames
        ])
        if self.hflip and self.random_crop and random.random() < 0.5:
            clip = torch.flip(clip, dims=(-1,))
        return clip
