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
import matplotlib.pyplot as plt
import cv2

def gradcam(model, dataloader, is_train, args):
    model.eval()
    model = model.to(args['device'])
    for x, y, file_name, dset_type in tqdm(dataloader, total=len(dataloader)):
        x = x.to(args['device'])
        pred = model(x)
        pred[:, torch.argmax(pred, dim=1)].backward()
        gradients = model.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = model.get_activations(x).detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze().cpu()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)

        img = cv2.imread(f'./data_edited/{dset_type[0]}_image/{file_name[0]}')
        heatmap = cv2.resize(np.array(heatmap), (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        save_dir = f'./gradcam/{is_train}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, f'{file_name[0]}.jpg'), superimposed_img)
