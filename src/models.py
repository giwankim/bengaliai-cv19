import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F


class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__[
                'resnet34'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)

        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)

    def forward(self, x):
        bs = x.shape[0]
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        out0 = self.l0(x)
        out1 = self.l1(x)
        out2 = self.l2(x)
        return out0, out1, out2


class SEResNext50(nn.Module):
    def __init__(self, pretrained):
        super(SEResNext50, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__[
                'se_resnext50_32x4d'](pretrained='imagenet')
        else:
            self.model = pretrainedm

        self.l0 = nn.Linear(2048, 168)
        self.l1 = nn.Linear(2048, 11)
        self.l2 = nn.Linear(2048, 7)

    def forward(self, x):
        batch_size = len(x)

        x = self.model.features(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2
