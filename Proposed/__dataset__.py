import os
import re
import cv2
import json
import functions
import numpy as np
import albumentations as A
import torchvision.transforms as transforms

from __syslog__ import EventlogHandler
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# Image sizes used by the SR network
# -------------------------------------------------

IMG_LR = (40, 20)
IMG_HR = (160, 80)

# -------------------------------------------------
# Build ICPR LR–HR pairs from track folders
# -------------------------------------------------

def build_icpr_samples(root_dir):
    """
    Walk ICPR dataset and build LR–HR pairs.

    Expected structure:

    root/
      Scenario-A/
        Country/
          track_xxxxx/
            lr-001.png ... lr-005.png
            hr-001.png ... hr-005.png
            annotations.json
    """

    samples = []
    root_dir = Path(root_dir)

    for scenario in root_dir.iterdir():
        if not scenario.is_dir():
            continue

        for region in scenario.iterdir():
            if not region.is_dir():
                continue

            for track in region.iterdir():
                if not track.is_dir():
                    continue

                ann = track / "annotations.json"
                if not ann.exists():
                    continue

                with open(ann) as f:
                    meta = json.load(f)

                plate = meta["plate"]
                layout = meta.get("layout", "unknown")
                tp = meta.get("type", "car")

                # Pair lr-i with hr-i
                for i in range(1, 6):
                    lr = track / f"lr-{i:03d}.png"
                    hr = track / f"hr-{i:03d}.png"

                    if lr.exists() and hr.exists():
                        samples.append({
                            "LR": lr.as_posix(),
                            "HR": hr.as_posix(),
                            "plate": plate,
                            "layout": layout,
                            "type": tp,
                        })

    print(f"[INFO] Loaded {len(samples)} LR–HR pairs from ICPR")

    return samples

# -------------------------------------------------
# PyTorch Dataset
# -------------------------------------------------

class customDataset(Dataset):

    def __init__(self, samples, augmentation=True):

        self.x = samples
        self.to_tensor = transforms.ToTensor()
        self.augmentation = augmentation

        # Used for padding plates to fixed aspect ratio
        self.background_color = (127, 127, 127)
        self.aspect_ratio = 2.0
        self.min_ratio = self.aspect_ratio - 0.15
        self.max_ratio = self.aspect_ratio + 0.15

        # HR augmentations
        self.transformHR = np.array([
            A.HueSaturationValue(20, 30, 20, always_apply=True),
            A.RandomBrightnessContrast(0.2, 0.2, always_apply=True),
            A.RandomGamma(gamma_limit=(80, 120), always_apply=True),
            None
        ])

        # LR augmentations
        self.transformLR = np.array([
            A.HueSaturationValue(20, 30, 20, always_apply=True),
            A.RandomBrightnessContrast(0.2, 0.2, always_apply=True),
            A.RandomGamma(gamma_limit=(80, 120), always_apply=True),
            None
        ])

        print("Dataset aspect ratio:", self.aspect_ratio)

    # -------------------------------------------------

    def open_image(self, path):
        img = Image.open(path)
        return np.array(img)

    # -------------------------------------------------

    def __len__(self):
        return len(self.x)

    # -------------------------------------------------

    def __getitem__(self, index):

        sample = self.x[index]

        imgHR = self.open_image(sample["HR"])
        imgLR = self.open_image(sample["LR"])

        # Random augmentation
        if self.augmentation:
            augHR = np.random.choice(self.transformHR)
            augLR = np.random.choice(self.transformLR)

            if augHR is not None:
                imgHR = augHR(image=imgHR)["image"]

            if augLR is not None:
                imgLR = augLR(image=imgLR)["image"]

        plate = sample["plate"]
        layout = sample["layout"]
        tp = sample["type"]

        # Pad to fixed ratio
        imgLR, _, _ = functions.padding(
            imgLR,
            self.min_ratio,
            self.max_ratio,
            color=self.background_color,
        )

        imgHR, _, _ = functions.padding(
            imgHR,
            self.min_ratio,
            self.max_ratio,
            color=self.background_color,
        )

        # Resize to network input sizes
        imgLR = cv2.resize(imgLR, IMG_LR, interpolation=cv2.INTER_CUBIC)
        imgHR = cv2.resize(imgHR, IMG_HR, interpolation=cv2.INTER_CUBIC)

        # Convert to tensors
        imgLR = self.to_tensor(imgLR)
        imgHR = self.to_tensor(imgHR)

        file_name = Path(sample["HR"]).name

        return {
            "LR": imgLR,
            "HR": imgHR,
            "plate": plate,
            "layout": layout,
            "type": tp,
            "file": file_name,
        }

# -------------------------------------------------
# DataLoader builder
# -------------------------------------------------

@EventlogHandler
def load_dataset(root, batch_size, mode, pin_memory, num_workers):

    all_samples = build_icpr_samples(root)

    # Phase-1: random split
    np.random.shuffle(all_samples)

    n = len(all_samples)
    train = all_samples[: int(0.9 * n)]
    val = all_samples[int(0.9 * n):]

    if mode in [0, 1]:

        train_dl = DataLoader(
            customDataset(train, augmentation=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_dl = DataLoader(
            customDataset(val, augmentation=False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_dl, val_dl

    elif mode == 2:

        test_dl = DataLoader(
            customDataset(all_samples, augmentation=False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return test_dl


print("Hi")
