import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from torch import optim
from torchvision.transforms import transforms
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import os
import pandas as pd
import numpy as np
import cv2
import argparse

try:
    import dataset
    import models
    import utils
    import grad_cam
except:
    from bind_classification import dataset
    from bind_classification import models
    from bind_classification import utils
    from bind_classification import grad_cam

def evaluate(model, dataloader, args):
    with torch.no_grad():
        model.eval()
        total = 0
        correct = 0
        
        label_0_total = 0
        label_0_correct = 0
        label_1_total = 0
        label_1_correct = 0
        wrong = []
        confidence = []
        for x, y, file_name, dset_type in dataloader:
            x = x.to(args.device)
            y = y.to(args.device)
            proba = F.softmax(model(x), dim=1)
            pred = torch.argmax(proba, dim=1)
            is_correct = pred==y
            total += x.shape[0]
            correct += is_correct.sum()
            
            label_0_total += (y==0).sum()
            label_1_total += (y==1).sum()
            label_0_correct += (y==0 & is_correct).sum()
            label_1_correct += (y==1 & is_correct).sum()
            
            for idx in range(is_correct.shape[0]):
                if is_correct[idx] == False:
                    wrong.append((file_name[idx], dset_type[idx]))
                    confidence.append(proba[idx, y[idx]].cpu().item())
    return total, correct.item(), label_0_total, label_0_correct, label_1_total, label_1_correct, wrong, confidence     


def inference(opt):
    import os
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    # os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_number
    
    utils.seed_everything(seed=opt.seed)
    root_dir = opt.img_dir
    tree_dir = list(os.walk(root_dir))
    folders = tree_dir[0][1]
    datasets = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(opt.image_size),
        transforms.Normalize(0.5, 0.5, 0.5)
    ])
        
    for folder in folders:
        img_dir = os.path.join(root_dir, folder)
        file_names = os.listdir(img_dir)
        file_names.sort()
        dset = dataset.CustomDataset(img_dir=img_dir,
                                    img_names=file_names,
                                    label=2,
                                    transform=transform,
                                    dset_type=folder)
        datasets.append(dset)
    
    collated_dataset = dataset.CollateDataset(images=[d.images for d in datasets],
                                            labels=[d.labels for d in datasets],
                                            img_names=[d.img_names for d in datasets],
                                            dset_types=[d.dset_type for d in datasets])

    dataloader = data.DataLoader(collated_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                drop_last=False)
    
    model = models.EfficientNetB3()
    model.load_state_dict(torch.load(opt.model_dir))
    model = model.to(opt.device)

    
    print(f"gpu -> {opt.n_gpu}")
    if torch.cuda.device_count() < opt.n_gpu:
        raise ValueError("Wrong n_gpu input")
    
    if opt.n_gpu > 1:
        model = nn.DataParallel(model)

    preds = []
    file_names = []
    dset_types = []

    with torch.no_grad():
        model.eval()
        for x, y, file_name, dset_type in tqdm(dataloader, total=len(dataloader)):
            x = x.to(opt.device)
            pred = model(x)
            pred = torch.argmax(pred, dim=1).cpu().detach()
            preds.extend([i.item() for i in pred])
            file_names.extend(file_name)
            dset_types.extend(dset_type)

    result = pd.DataFrame()
    result['folder'] = pd.Series(dset_types)
    result['img_path'] = pd.Series(file_names)
    result['predictions'] = pd.Series(preds)
    
    if not os.path.exists(opt.save_dir):
        print(f'Directory created. {opt.save_dir}')
        os.makedirs(opt.save_dir)
    save_dir = os.path.join(opt.save_dir, "result.csv")
    result.to_csv(save_dir, index=False)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_dir', type=str)           
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--gpu_number', type=str, default="0")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--image_size', type=tuple, default=(300, 300))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    
    opt = parser.parse_args()
    model = inference(opt)
    
    
    