#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Written by yangmin09

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import core.net as net
from core.config import cfg
from model.resnet import ResNet, ResHead


class Delg(nn.Module):
    def __init__(self):
        super(Delg, self).__init__()
        self.globalmodel = ResNet()
        self.desc_cls = Arcface(cfg.MODEL.HEADS.REDUCTION_DIM, cfg.MODEL.NUM_CLASSES)
        self.localmodel = SpatialAttention2d(1024)
        self.att_cls = ResHead(512, cfg.MODEL.NUM_CLASSES)
    
    def forward(self, x, targets):
        global_feature, feamap = self.globalmodel(x)
        global_logits = self.desc_cls(global_feature, targets)
        
        block3 = feamap.detach()
        local_feature, att_score = self.localmodel(block3)
        local_logits = self.att_cls(local_feature)
        return global_feature, global_logits, local_feature, local_logits, att_score


class SpatialAttention2d(nn.Module):
    '''
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    '''
    def __init__(self, in_c, act_fn='relu'):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 512, 1, 1)
        self.bn = nn.BatchNorm2d(512, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(512, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20) # use default setting.

        for conv in [self.conv1, self.conv2]: 
            conv.apply(net.init_weights)

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        x = self.conv1(x)
        x = self.bn(x)
        
        feature_map_norm = F.normalize(x, p=2, dim=1)
         
        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        
        x = att * feature_map_norm
        return x, att_score
    
    def __repr__(self):
        return self.__class__.__name__


class Arcface(nn.Module):
    def __init__(self, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = cfg.MODEL.HEADS.SCALE
        self._m = cfg.MODEL.HEADS.MARGIN

        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        self.register_buffer('t', torch.zeros(1))

    def forward(self, features, targets):
        # get cos(theta)
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta * self._s
        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )

