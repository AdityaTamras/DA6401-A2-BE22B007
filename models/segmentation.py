import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder

def _make_decoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    
class VGG11UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, dropout_p=0.5):
        super(VGG11UNet, self).__init__()
        self.encoder=VGG11Encoder(in_channels=in_channels)

        self.upsample1=nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv1=_make_decoder_block(1024, 512)

        self.upsample2=nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2=_make_decoder_block(512, 256)

        self.upsample3=nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3=_make_decoder_block(256, 128)

        self.upsample4=nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4=_make_decoder_block(128, 64)

        self.upsample5=nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.output_conv=nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        encoded, feature_dict=self.encoder(x, return_features=True)

        d1=self.upsample1(encoded)
        d1=torch.cat([d1, feature_dict['block4']], dim=1)
        d1=self.conv1(d1)

        d2=self.upsample2(d1)
        d2=torch.cat([d2, feature_dict['block3']], dim=1)
        d2=self.conv2(d2)

        d3=self.upsample3(d2)
        d3=torch.cat([d3, feature_dict['block2']], dim=1)
        d3=self.conv3(d3)

        d4=self.upsample4(d3)
        d4=torch.cat([d4, feature_dict['block1']], dim=1)
        d4=self.conv4(d4)

        d5=self.upsample5(d4)

        output=self.output_conv(d5)
        return output

