import torch
import torch.nn as nn
from models.convnext import ConvNeXt3dSTOIC, LayerNorm3d
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

class UperNeXtDecoder(nn.Module):
    def __init__(self, modelconfig, config):
        super(UperNeXtDecoder, self).__init__()
        dims=[96, 192, 384, 768] if modelconfig.size != 'micro' else [64, 128, 256, 512]
        upsample_dim = 16

        self.upsample_blocks = nn.ModuleList()

        upsample_block1 = nn.Sequential(
            nn.Conv3d(dims[-1], upsample_dim, 1), nn.ReLU(),
            LayerNorm3d(normalized_shape=upsample_dim, data_format='channels_first'),
            UpsampleLayer(factor=2 ** 3),
        )
        self.upsample_blocks.append(upsample_block1)

        upsample_block2 = nn.Sequential(
            nn.Conv3d(dims[-2], upsample_dim, 1), nn.ReLU(),
            LayerNorm3d(normalized_shape=upsample_dim, data_format='channels_first'),
            UpsampleLayer(factor=2**2),
        )
        self.upsample_blocks.append(upsample_block2)

        upsample_block3 = nn.Sequential(
            nn.Conv3d(dims[-3], upsample_dim, 1), nn.ReLU(),
            LayerNorm3d(normalized_shape=upsample_dim, data_format='channels_first'),
            UpsampleLayer(factor=2**1),
        )
        self.upsample_blocks.append(upsample_block3)

        upsample_block4 = nn.Sequential(
            nn.Conv3d(dims[-4], upsample_dim, 1), nn.ReLU(),
            LayerNorm3d(normalized_shape=upsample_dim, data_format='channels_first'),
            UpsampleLayer(factor=2**0),
        )
        self.upsample_blocks.append(upsample_block4)


        self.segmentation_head = nn.Sequential(
            nn.Conv3d(4*upsample_dim, upsample_dim, 7, padding='same'),
            nn.ReLU(), #LayerNorm3d(normalized_shape=upsample_dim, data_format='channels_first'),
            UpsampleLayer(factor=4),
            nn.Conv3d(upsample_dim, 1, 3, padding='same'),
                                          )

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias != None:
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        outputs = list()
        for i, feature in enumerate(features):
            upsample_block = self.upsample_blocks[-i-1]
            outputs.append(upsample_block(feature))
        output = torch.cat(outputs, dim=1)
        output = self.segmentation_head(output)
        return output


class UpsampleLayer(nn.Module):
    def __init__(self, factor=2):
        super(UpsampleLayer, self).__init__()
        self.factor = factor

    def forward(self, x):
        B, C, D, H, W = x.shape
        if W==3 and H==3 and D==3:
            size = (D*self.factor+self.factor//2, H*self.factor+self.factor//2, W*self.factor+self.factor//2)
            x = torch.nn.functional.interpolate(x, size=size, mode='trilinear', align_corners=True)
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode='trilinear', align_corners=True)
        return x

class UperNeXt(nn.Module):
    def __init__(self, modelconfig, config):
        super(UperNeXt, self).__init__()
        self.encoder = ConvNeXt3dSTOIC(modelconfig, config)
        self.decoder = UperNeXtDecoder(modelconfig, config)

    def forward(self, x):
        features = self.encoder(x, age=None, sex=None, get_features=True)
        output = self.decoder(features)
        return output


class MultiNeXt(nn.Module):
    def __init__(self, modelconfig, config):
        super(MultiNeXt, self).__init__()
        self.encoder = ConvNeXt3dSTOIC(modelconfig, config)
        self.decoder = UperNeXtDecoder(modelconfig, config)

        dim = 768 if modelconfig.size != 'micro' else 512
        self.pre_head_mos = nn.Linear(dim, dim)
        self.head_mos = nn.Linear(dim, 2)
        self.norm_mos = nn.LayerNorm(dim, eps=1e-6)

        self.pre_head_hust = nn.Linear(dim, dim)
        self.head_hust = nn.Linear(dim, 2)
        self.norm_hust = nn.LayerNorm(dim, eps=1e-6)
        self.metadata_prehead_hust = nn.Linear(4, dim, bias=False)

    # mode can be 'all', 'stoic', 'mosmed', or 'tcia'
    def forward(self, x, age=None, sex=None, mode='all'):
        features = self.encoder(x, age=age, sex=sex, get_features=True)

        #classification
        if mode == 'classification' or mode == 'all' or mode=='stoic':
            x = features[-1]
            x = self.encoder.main_model.norm(x.mean([-3, -2, -1]))
            x = self.encoder.main_model.pre_head(x)
            if (age is not None) and (sex is not None):
                metadata = torch.cat((sex.float(), age.float().unsqueeze(1)), dim=1)
                metadata = self.encoder.main_model.metadata_prehead(metadata)
                x = F.relu(x + metadata)
            else:
                x = F.relu(x)
            x = self.encoder.main_model.head(x)
        else:
            x = None

        if mode == 'mosmed' or mode == 'all':
            x2 = features[-1]
            x2 = self.norm_mos(x2.mean([-3, -2, -1]))
            x2 = self.pre_head_mos(x2)
            x2 = F.relu(x2)
            x2 = self.head_mos(x2)
        else:
            x2 = None

        if mode == 'hust' or mode == 'all':
            x3 = features[-1]
            x3 = self.norm_mos(x3.mean([-3, -2, -1]))
            x3 = self.pre_head_hust(x3)
            if (age is not None) and (sex is not None):
                metadata = torch.cat((sex.float(), age.float().unsqueeze(1)), dim=1)
                metadata = self.metadata_prehead_hust(metadata)
                x3 = F.relu(x3 + metadata)
            else:
                x3 = F.relu(x3)
            x3 = self.head_hust(x3)
        else:
            x3 = None

        #segmentation
        output = self.decoder(features) if mode == 'segmentation' or mode == 'all' or mode == 'tcia' else None

        return x, x2, x3, output