import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class ClassificationHead(nn.Module):
    def __init__(self, num_classes=37, dropout_p=0.5):
        super(ClassificationHead, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d((7, 7))
        self.classifier=nn.Sequential(
            nn.Linear(25088, 4096),
            nn.BatchNorm1d(4096),
            CustomDropout(dropout_p),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            CustomDropout(dropout_p),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        out=self.avgpool(x)
        out=torch.flatten(out, 1)
        out=self.classifier(out)
        return out
    
class VGG11Classifier(nn.Module):
    def __init__(self, num_classes=37, in_channels=3, dropout_p=0.5):
        super(VGG11Classifier, self).__init__()
        self.encoder=VGG11Encoder(in_channels=in_channels)
        self.head=ClassificationHead(num_classes=num_classes, dropout_p=dropout_p)

    def forward(self, x):
        encoded=self.encoder(x, return_features=False)
        return self.head(encoded)

