from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

import torch
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.layers = [layer for layer in self.model.children()]
        self.model = nn.Sequential(
            *self.layers[:-1]
        )
        self.classifier = nn.Linear(2048, 2)

    def forward(self, x):
        out = self.model(x).squeeze(2).squeeze(2)
        out = self.classifier(out)
        return out
    

class ResNet50GradCam(nn.Module):
    def __init__(self):
        super(ResNet50GradCam, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.layers = [layer for layer in self.model.children()]
        self.model = nn.Sequential(
            *self.layers[:-2]
        )
        self.pooling = self.layers[-2]
        self.classifier = nn.Linear(2048, 2)
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.model(x)

    def forward(self, x):
        out = self.model(x)
        if out.requires_grad:
            h = out.register_hook(self.activations_hook)
        out = self.pooling(out).squeeze(2).squeeze(2)
        out = self.classifier(out)
        return out
    

class EfficientNetB3(nn.Module):
    def __init__(self):
        super(EfficientNetB3, self).__init__()
        self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.layers = [layer for layer in self.model.children()]
        self.model = nn.Sequential(
            *self.layers[:-1]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1536, 2)
        )

    def forward(self, x):
        out = self.model(x).squeeze(2).squeeze(2)
        out = self.classifier(out)
        return out
    

class EfficientNetB3Gradcam(nn.Module):
    def __init__(self):
        super(EfficientNetB3Gradcam, self).__init__()
        self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.layers = [layer for layer in self.model.children()]
        self.model = nn.Sequential(
            *self.layers[:-1]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1536, 2)
        )

    def forward(self, x):
        out = self.model(x).squeeze(2).squeeze(2)
        out = self.classifier(out)
        return out

    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.model(x)

    def forward(self, x):
        out = self.model(x)
        if out.requires_grad:
            h = out.register_hook(self.activations_hook)
        out = self.pooling(out).squeeze(2).squeeze(2)
        out = self.classifier(out)
        return out
    