import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from skimage import color
from PIL import Image
import config

class ColorizationDataset(Dataset):
    """
    Wraps Places365 (or STL-10) and returns (L_channel, AB_channels)
    pairs for training the colorization CNN.
    """

    def __init__(self, split="val", size=config.IMAGE_SIZE):
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),   # just for loading; we convert to LAB manually
        ])

        if config.DATASET == "places365":
            self.dataset = datasets.Places365(
                root=config.DATA_DIR,
                split=split,          # "val" for dev, "train-standard" for full training
                small=True,           # 256x256 version
                download=True,
                transform=None        # we handle transforms ourselves
            )
        elif config.DATASET == "stl10":
            self.dataset = datasets.STL10(
                root=config.DATA_DIR,
                split="train" if split != "val" else "test",
                download=True,
                transform=None
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]

        # Convert PIL → numpy RGB (0–255) → resize → LAB
        img = img.convert("RGB").resize((self.size, self.size))
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_lab = color.rgb2lab(img_np)  # L: [0,100], AB: [-128, 127]

        # L channel → grayscale input, normalized to [-1, 1]
        L  = (img_lab[:, :, 0:1] / 50.0) - 1.0          # (H, W, 1)
        AB = img_lab[:, :, 1:3] / config.AB_NORM          # (H, W, 2)

        # → tensors (C, H, W)
        L  = torch.tensor(L).permute(2, 0, 1).float()
        AB = torch.tensor(AB).permute(2, 0, 1).float()

        return L, AB


def get_dataloader(split="val"):
    dataset = ColorizationDataset(split=split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(split != "val"),
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    return loader