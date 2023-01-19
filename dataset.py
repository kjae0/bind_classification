import torch
from torch.utils import data
import pandas as pd
import numpy as np

import os
from tqdm.auto import tqdm
from PIL import Image

class CustomDataset(data.Dataset):
    def __init__(self, img_dir, img_names, label, transform, dset_type=None):
        self.images = []
        self.img_names = img_names
        self.dset_type = [dset_type for _ in range(len(img_names))]
        
        for d in tqdm(img_names, total=len(img_names)):
            self.images.append(transform(Image.open(os.path.join(img_dir, d)).convert('RGB')))
        self.labels = [label for i in range(len(img_names))]
        
        print(f'Dataset size is image : {len(self.images)}, label : {len(self.labels)}.')
        
    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.img_names[index], self.dset_type[index]
    
    def __len__(self):
        return len(self.images)

class CollateDataset(data.Dataset):
    def __init__(self, images, labels, img_names, dset_types) -> None:
        self.collated_images = []
        self.collated_labels = []
        self.collated_img_names = []
        self.collated_dset_type = []

        for idx in range(len(images)):
            if len(images[idx]) != len(labels[idx]):
                print("Warning! Length of images and labels are different!")
            self.collated_images.extend(images[idx])
            self.collated_labels.extend(labels[idx])
            self.collated_img_names.extend(img_names[idx])
            self.collated_dset_type.extend(dset_types[idx])
        
        print(f'Collated dataset size is image : {len(self.collated_images)}\nlabel : {len(self.collated_labels)}\nimg names : {len(self.collated_img_names)}\ndset types : {len(self.collated_dset_type)}')
    
    def __getitem__(self, index):
        return self.collated_images[index], self.collated_labels[index], self.collated_img_names[index], self.collated_dset_type[index]
    
    def __len__(self):
        return len(self.collated_images)