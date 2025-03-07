import torch
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate
import os
from PIL import Image
import pandas as pd



class ImageClassificationDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.entries = os.listdir(img_dir)
        self.transform = transform
    
    
    def __len__(self):
        return len(self.entries)
    
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.entries[idx])
        erosion_score, jsn_score = os.path.splitext(self.entries[idx])[0].split("_")[-2:]

        img = Image.open(img_path)
        img = self.transform(img)
        return img, int(erosion_score), int(jsn_score)


class EvalImageDataset(Dataset):
    def __init__(self, image_dir, bbox_csv, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.bbox_df = pd.read_csv(bbox_csv)
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for _, row in self.bbox_df.iterrows():
            patient_id = row["patient_id"]
            joint_id = row["joint_id"]
            image_name = f"{int(patient_id)}_{int(joint_id)}.jpeg"
            image_path = os.path.join(self.image_dir, image_name)
            if os.path.exists(image_path):
                samples.append((image_path, patient_id, joint_id, row["xcenter"], row["ycenter"], row["dx"], row["dy"]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, patient_id, joint_id, xcenter, ycenter, dx, dy = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "patient_id": patient_id,
            "joint_id": joint_id,
            "xcenter": xcenter,
            "ycenter": ycenter,
            "dx": dx,
            "dy": dy,
        }


def initialize_data(cfg):
    train_dataset = instantiate(cfg.train_dataset)
    val_dataset = instantiate(cfg.val_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers
    )
    return train_loader, val_loader