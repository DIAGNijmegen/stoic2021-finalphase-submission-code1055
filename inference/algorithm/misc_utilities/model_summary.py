# -*- coding: utf-8 -*-
"""
Created on 28.01.22

@author: ludwikat

"""
from config.modelconfig import ConvNeXt3DConfig
from models.convnext import ConvNeXt3dSTOIC

from torchsummary import summary

from models.hypernet import HyperConvNeXt3d

model = ConvNeXt3dSTOIC(ConvNeXt3DConfig()).cuda()

summary(model, (1, 128, 128, 128))
print("-----------------------------------------------------------------------")

model = HyperConvNeXt3d(in_chans=1, num_classes=2).cuda()
summary(model, (1, 128, 128, 128))