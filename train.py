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

def train(opt):
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_number
    
    utils.seed_everything(seed=opt.seed)
    root_dir = opt.img_dir
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
        transforms.Resize(opt.image_size),
        transforms.Normalize(0.5, 0.5, 0.5)
    ])

    good_dataset = dataset.CustomDataset(img_dir=good_dir,
                                        img_names=good_files,
                                        label=1,
                                        transform=transform,
                                        dset_type='good_image')
    bad_dataset = dataset.CustomDataset(img_dir=bad_dir,
                                        img_names=bad_files,
                                        label=0,
                                        transform=transform,
                                        dset_type='bad_image')
    ab_dataset = dataset.CustomDataset(img_dir=ab_dir,
                                        img_names=ab_files,
                                        label=0,
                                        transform=transform,
                                        dset_type='ab_image')
    dsets = [good_dataset, bad_dataset, ab_dataset]

    collated_dataset = dataset.CollateDataset(images=[d.images for d in dsets],
                                            labels=[d.labels for d in dsets],
                                            img_names=[d.img_names for d in dsets],
                                            dset_types=[d.dset_type for d in dsets])

    train_dataset, val_dataset = train_test_split(collated_dataset, 
                                test_size=opt.test_size,
                                shuffle=True,
                                random_state=opt.seed,
                                stratify=collated_dataset.collated_labels)

    train_dataloader = data.DataLoader(train_dataset,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                drop_last=True)

    val_dataloader = data.DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                drop_last=False)

    if opt.model == 'resnet50':
        if opt.grad_cam:
            model = models.ResNet50GradCam().to(opt.device)
        else:
            model = models.ResNet50().to(opt.device)
    elif opt.model == 'efficientnetb3':
        if opt.grad_cam:
            model = models.EfficientNetB3Gradcam().to(opt.device)
        else:
            model = models.EfficientNetB3().to(opt.device)

    print(f"gpu -> {opt.n_gpu}")
    if torch.cuda.device_count() < opt.n_gpu:
        raise ValueError("Wrong n_gpu input")
    
    if opt.multi_gpu:
        model = nn.DataParallel(model)
    
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False
    )
    
    
    if opt.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    else:
        raise ValueError("wrong optimizer input")
    
    if opt.scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    else:
        raise ValueError("wrong scheduler input")
        
    if opt.criterion == 'cross_entropy_loss':
        criterion = nn.CrossEntropyLoss()
    elif opt.criterion == 'ldam_loss':
        criterion = utils.LDAMLoss([len(ab_dataset)+len(bad_dataset), len(good_dataset)])
    else:
        raise ValueError("wrong criterion input")
    
    wrong_imgs = []

    for epoch in range(opt.n_epochs):
        losses = 0
        n_iter = 0
        for x, y, _, _ in tqdm(train_dataloader, total=len(train_dataloader)):
            model.train()
            x = x.to(opt.device).float()
            y = y.to(opt.device).long()

            pred = model(x)
            loss = criterion(pred, y)
            
            losses += loss
            n_iter += 1
            
            optimizer.zero_grad()        
            loss.backward()
            optimizer.step()
            
        print(f'{epoch+1} loss : {losses/n_iter}')
        train_total, train_correct, train_0_total, train_0_correct, train_1_total, train_1_correct, _, _ = evaluate(model, train_dataloader, args=opt)
        val_total, val_correct, val_0_total, val_0_correct, val_1_total, val_1_correct, wrong, confidence = evaluate(model, val_dataloader, args=opt)
        print(f'train accuracy : {train_correct/train_total}')
        print(f'train 0 class accuracy : {train_0_correct/train_0_total}')
        print(f'train 1 class accuracy : {train_1_correct/train_1_total}')
        print(f'val accuracy : {val_correct/val_total}')
        print(f'val 0 class accuracy : {val_0_correct/val_0_total}')
        print(f'val 1 class accuracy : {val_1_correct/val_1_total}')
        wrong_imgs.append(wrong)

        if opt.save_log:
            save_dir = f"./wrong_imgs/{opt.name}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            with open(os.path.join(save_dir, f"wrong_imgs.txt"), 'w') as f:
                for idx, wrong in enumerate(wrong_imgs):
                    f.write(f"epoch {idx+1}=====================================\n")
                    for (img_name, dset), conf in zip(wrong, confidence):
                        f.write(dset+" ")
                        f.write(img_name+" ")
                        f.write(str(conf))
                        f.write("\n")
                    f.write("\n")
                
    if opt.save_model:
        model_dir = f"./saved_model/{opt.name}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model, os.path.join(model_dir, f"{opt.n_epochs}_{opt.name}.pt"))

    if opt.grad_cam:
        train_dataloader = data.DataLoader(train_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    drop_last=False)

        val_dataloader = data.DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    drop_last=False)

        grad_cam.gradcam(model, train_dataloader, is_train=f'{opt.name}_train', args=opt)
        grad_cam.gradcam(model, val_dataloader, is_train=f'{opt.name}_validation', args=opt)

    return model

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_dir', type=str)           
    parser.add_argument('--batch_size', type=int, default=32)      
    parser.add_argument('--criterion', type=str, default='cross_entropy_loss')  
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scheduler', type=boolean_string, default=True)
    parser.add_argument('--optimizer', type=str, default='adam'),
    parser.add_argument('--n_gpu', type=int, default=1),
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--image_size', type=tuple, default=(300, 300))
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--save_log', type=boolean_string, default=True)
    parser.add_argument('--save_model', type=boolean_string, default=True)
    parser.add_argument('--grad_cam', type=boolean_string, default=False)
    parser.add_argument('--name', type=str)
    parser.add_argument('--multi_gpu', type=boolean_string, default=False)
    parser.add_argument('--gpu_number', type=str, default="0")
    parser.add_argument('--model', type=str)
    
    opt = parser.parse_args()
    model = train(opt)
    model_dir = f"./saved_model/{opt.name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, f"{opt.n_epochs}_{opt.name}.pt"))
    
    