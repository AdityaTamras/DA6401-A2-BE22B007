import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout

class _RegressionHead(nn.Module):
    def __init__(self, dropout_p=0.5):
      super(_RegressionHead, self).__init__()
      self.avgpool=nn.AdaptiveAvgPool2d((7, 7))
      self.regressor=nn.Sequential(
         nn.Linear(25088, 1024),
         nn.ReLU(inplace=True),
         CustomDropout(dropout_p),
         nn.Linear(1024, 4),
         nn.Sigmoid()
      )
    
    def forward(self, x):
      out=self.avgpool(x)
      out=torch.flatten(out, 1)
      out=self.regressor(out)
      return out

class VGG11Localizer(nn.Module):
    def __init__(self, in_channels=3, dropout_p=0.5):
      super(VGG11Localizer, self).__init__()
      self.encoder=VGG11Encoder(in_channels=in_channels)
      self.head=_RegressionHead(dropout_p=dropout_p)
      self.image_size=(224, 224)

    def forward(self, x):
      encoded=self.encoder(x, return_features=False)
      normalized=self.head(encoded)
      H, W = self.image_size
      scale=torch.tensor([W, H, W, H], dtype=normalized.dtype, device=normalized.device)
      pixel_boxes=normalized*scale
      return pixel_boxes
      

