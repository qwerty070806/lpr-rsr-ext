import os
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

IMG_LR = (64, 32)
IMG_HR = (256, 128)

# -------------------------------------------------
# Crop using provided corners
# -------------------------------------------------

def normalize_corner_keys(corners):

    fixed = {}

    for k, v in corners.items():

        # fix .png.png bug
        if k.endswith(".png.png"):
            k = k.replace(".png.png", ".png")

        fixed[k] = v

    return fixed


def crop_from_corners(img, corners):

    pts = np.array([
        corners["top-left"],
        corners["top-right"],
        corners["bottom-right"],
        corners["bottom-left"],
    ])

    x_min = int(min(p[0] for p in pts))
    y_min = int(min(p[1] for p in pts))
    x_max = int(max(p[0] for p in pts))
    y_max = int(max(p[1] for p in pts))

    return img[y_min:y_max, x_min:x_max]

# -------------------------------------------------
# Build ICPR LR–HR pairs
# -------------------------------------------------

def build_icpr_samples(root_dir):

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

                # ✅ CORRECT ICPR FIELDS
                plate = meta["plate_text"]
                layout = meta["plate_layout"]
                tp = "car"

                corners = normalize_corner_keys(meta.get("corners", {}))

                for i in range(1, 6):
                
                    lr = track / f"lr-{i:03d}.png"
                    hr = track / f"hr-{i:03d}.png"
                
                    if not (lr.exists() and hr.exists()):
                        continue
                
                    if lr.name not in corners or hr.name not in corners:
                        print(f"[WARN] Missing corners in {track.name} for {lr.name}")
                        continue
                
                    samples.append({
                        "LR": lr.as_posix(),
                        "HR": hr.as_posix(),
                        "plate": plate,
                        "layout": layout,
                        "type": tp,
                        "corners_lr": corners[lr.name],
                        "corners_hr": corners[hr.name],
                    })

    print(f"[INFO] Loaded {len(samples)} LR–HR pairs from ICPR")

    return samples

# -------------------------------------------------
# Dataset
# -------------------------------------------------

class customDataset(Dataset):

    def __init__(self, samples, augmentation=True):

        self.x = samples
        self.to_tensor = transforms.ToTensor()
        self.augmentation = augmentation

        self.background_color = (127, 127, 127)
        self.aspect_ratio = 2.0
        self.min_ratio = self.aspect_ratio - 0.15
        self.max_ratio = self.aspect_ratio + 0.15

        self.transformHR = np.array([
            A.HueSaturationValue(20, 30, 20, always_apply=True),
            A.RandomBrightnessContrast(0.2, 0.2, always_apply=True),
            A.RandomGamma(gamma_limit=(80, 120), always_apply=True),
            None
        ])

        self.transformLR = np.array([
            A.HueSaturationValue(20, 30, 20, always_apply=True),
            A.RandomBrightnessContrast(0.2, 0.2, always_apply=True),
            A.RandomGamma(gamma_limit=(80, 120), always_apply=True),
            None
        ])

        print("Dataset aspect ratio:", self.aspect_ratio)

    # -------------------------------------------------

    def open_image(self, path):
        return np.array(Image.open(path))

    # -------------------------------------------------

    def __len__(self):
        return len(self.x)

    # -------------------------------------------------

    def __getitem__(self, index):

        sample = self.x[index]

        imgLR = self.open_image(sample["LR"])
        imgHR = self.open_image(sample["HR"])

        # crop plates
        imgLR = crop_from_corners(imgLR, sample["corners_lr"])
        imgHR = crop_from_corners(imgHR, sample["corners_hr"])

        # augmentation
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

        # padding
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

        # resize
        imgLR = cv2.resize(imgLR, IMG_LR, interpolation=cv2.INTER_CUBIC)
        imgHR = cv2.resize(imgHR, IMG_HR, interpolation=cv2.INTER_CUBIC)

        # tensors
        imgLR = self.to_tensor(imgLR)
        imgHR = self.to_tensor(imgHR)

        return {
            "LR": imgLR,
            "HR": imgHR,
            "plate": plate,
            "layout": layout,
            "type": tp,
            "file": Path(sample["HR"]).name,
        }

# -------------------------------------------------
# Loader
# -------------------------------------------------

@EventlogHandler
def load_dataset(root, batch_size, mode, pin_memory, num_workers):

    all_samples = build_icpr_samples(root)

    np.random.shuffle(all_samples)

    n = len(all_samples)
    train = all_samples[: int(0.9 * n)]
    val = all_samples[int(0.9 * n):]

    if mode in [0, 1]:

        return (
            DataLoader(customDataset(train, True), batch_size, True, num_workers=num_workers, pin_memory=pin_memory),
            DataLoader(customDataset(val, False), batch_size, False, num_workers=num_workers, pin_memory=pin_memory),
        )

    else:

        return DataLoader(
            customDataset(all_samples, False),
            batch_size,
            False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
