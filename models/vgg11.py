from typing import Dict, Tuple, Union
import torch
import torch.nn as nn


def _make_conv_block(in_channels, out_channels, num_convs):
    layers=[]
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels=out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class VGG11Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(VGG11Encoder, self).__init__()

        self.block1=_make_conv_block(in_channels, 64, 1)
        self.block2=_make_conv_block(64, 128, 1)
        self.block3=_make_conv_block(128, 256, 2)
        self.block4=_make_conv_block(256, 512, 2)
        self.block5=_make_conv_block(512, 512, 2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        s1=self.block1(x)
        s2=self.block2(s1)
        s3=self.block3(s2)
        s4=self.block4(s3)
        s5=self.block5(s4)
        if return_features:
            feature_dict={
                'block1': s1,
                'block2': s2,
                'block3': s3,
                'block4': s4
            }
            return s5, feature_dict
        return s5



