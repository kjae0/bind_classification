import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from torch import optim
from torchvision.transforms import transforms
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import os
import pandas as pd
import numpy as np
import dataset
import models
import utils
import cv2
import grad_cam

utils.seed_everything(seed=42)

def evaluate(model, dataloader):
    with torch.no_grad():
        model.eval()
        total = 0
        correct = 0
        wrong = []
        for x, y, file_name, dset_type in dataloader:
            x = x.to(args['device'])
            y = y.to(args['device'])
            pred = model(x)
            pred = torch.argmax(pred, dim=1)
            is_correct = pred==y
            total += x.shape[0]
            correct += is_correct.sum()
            for idx in range(is_correct.shape[0]):
                if is_correct[idx] == False:
                    wrong.append((file_name[idx], dset_type[idx]))
    return total, correct.item(), wrong
            
    
            
args = {'batch_size' : 16,
        'n_epochs' : 50,
        'lr' : 1e-3,
        'device' : 'cuda' if torch.cuda.is_available() else 'cpu'}


root_dir = "./data_edited/"
ab_dir = os.path.join(root_dir, "ab_image")
bad_dir = os.path.join(root_dir, "bad_image")
good_dir = os.path.join(root_dir, "good_image")

ab_files = os.listdir(ab_dir)
bad_files = os.listdir(bad_dir)
good_files = os.listdir(good_dir)

ab_files.sort()
bad_files.sort()
good_files.sort()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((300, 300)),
    transforms.Normalize(0.5, 0.5, 0.5)
])

good_dataset = dataset.CustomDataset(img_dir=good_dir,
                                     img_names=good_files,
                                     label=1,
                                     transform=transform,
                                     dset_type='good')
bad_dataset = dataset.CustomDataset(img_dir=bad_dir,
                                     img_names=bad_files,
                                     label=0,
                                     transform=transform,
                                     dset_type='bad')
ab_dataset = dataset.CustomDataset(img_dir=ab_dir,
                                     img_names=ab_files,
                                     label=0,
                                     transform=transform,
                                     dset_type='ab')
dsets = [good_dataset, bad_dataset, ab_dataset]

collated_dataset = dataset.CollateDataset(images=[d.images for d in dsets],
                                          labels=[d.labels for d in dsets],
                                          img_names=[d.img_names for d in dsets],
                                          dset_types=[d.dset_type for d in dsets])

train, val = train_test_split(collated_dataset, 
                              test_size=0.2,
                              shuffle=True,
                              random_state=42,
                              stratify=collated_dataset.collated_labels)

train_dataloader = data.DataLoader(train,
                             batch_size=args['batch_size'],
                             shuffle=True,
                             drop_last=True)

val_dataloader = data.DataLoader(val,
                             batch_size=args['batch_size'],
                             shuffle=False,
                             drop_last=False)

model = models.ResNet50GradCam().to(args['device'])
n_devices = torch.cuda.device_count()
print(f"gpu -> {n_devices}")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
