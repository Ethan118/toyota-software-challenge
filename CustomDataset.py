from torch.utils.data import Dataset
from torchvision.io import read_image

import cv2

import os
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, image_paths, annotations, transform=None, target_transform=None):
        self.img_dir = image_paths
        self.labels = annotations
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, index):
        image_names = os.listdir(self.img_dir)
        image = read_image(os.path.join(self.img_dir, image_names[index]))
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
