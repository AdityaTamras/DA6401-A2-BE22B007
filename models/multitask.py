import torch
import torch.nn as nn
import gdown
from models.vgg11 import VGG11Encoder
from models.classification import ClassificationHead
from models.localization import _RegressionHead
from models.segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_classes=37, seg_classes=3, in_channels=3, classifier_path="classifier.pth", localizer_path="localizer.pth", unet_path="unet.pth"):
        super(MultiTaskPerceptionModel, self).__init__()

        gdown.download(id="14yDx23jUnekXFvECvIjo46EYKzYDJ3Rr", output=classifier_path, quiet=False)
        gdown.download(id="1MkrNleTgwtdU-w0_DrVLF5G3vRvx4_Z7", output=localizer_path, quiet=False)
        gdown.download(id="1Q-WTdHMNLzunc492w7dZvKaM_K0UNqgZ", output=unet_path, quiet=False)

        self.encoder=VGG11Encoder(in_channels=in_channels)
        self.cls_head=ClassificationHead(num_classes=num_classes)
        self.reg_head=_RegressionHead()
        self.image_size=(224, 224)

        temp_unet=VGG11UNet(num_classes=seg_classes, in_channels=in_channels)
        self.upsample1=temp_unet.upsample1
        self.conv1=temp_unet.conv1
        self.upsample2=temp_unet.upsample2
        self.conv2=temp_unet.conv2
        self.upsample3=temp_unet.upsample3
        self.conv3=temp_unet.conv3
        self.upsample4=temp_unet.upsample4
        self.conv4=temp_unet.conv4
        self.upsample5=temp_unet.upsample5
        self.output_conv=temp_unet.output_conv

        self._load_weights(classifier_path, localizer_path, unet_path)

    def _load_weights(self, classifier_path, localizer_path, unet_path):
        cls_sd=torch.load(classifier_path, map_location='cpu')
        encoder_sd={
            k.replace('encoder.', ''): v for k, v in cls_sd.items() if k.startswith('encoder.')
        }
        self.encoder.load_state_dict(encoder_sd, strict=True)
        head_sd={
            k.replace('head.', ''): v for k, v in cls_sd.items() if k.startswith('head.')
        }
        self.cls_head.load_state_dict(head_sd, strict=True)

        loc_sd=torch.load(localizer_path, map_location='cpu')
        reg_sd={
            k.replace('head.', ''): v for k, v in loc_sd.items() if k.startswith('head.')
        }
        self.reg_head.load_state_dict(reg_sd, strict=True)

        unet_sd=torch.load(unet_path, map_location='cpu')
        decoder_modules=['upsample1', 'conv1', 'upsample2', 'conv2', 'upsample3', 'conv3', 'upsample4', 'conv4', 'upsample5', 'output_conv']
        for name in decoder_modules:
            module=getattr(self, name)
            sd={k.replace(f'{name}.', ''):v for k, v in unet_sd.items() if k.startswith(f'{name}.')}
            module.load_state_dict(sd, strict=True)

    def forward(self, x):

        encoded, feature_dict = self.encoder(x, return_features=True)
        cls_out=self.cls_head(encoded)

        norm_boxes=self.reg_head(encoded)
        H, W = self.image_size
        scale=torch.tensor([W, H, W, H], dtype=norm_boxes.dtype, device=norm_boxes.device)
        loc_out=norm_boxes*scale

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
        seg_out=self.output_conv(d5)

        return{
            'classification': cls_out,
            'localization': loc_out,
            'segmentation': seg_out
        }



        

