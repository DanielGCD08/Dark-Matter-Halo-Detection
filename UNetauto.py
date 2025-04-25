import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from pylint.checkers.spelling import dict_choices
from torchvision.models.vgg import make_layers
from collections import OrderedDict


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target):
        # 输入形状:
        #   pred  -> [N, 1, H, W] （经过Sigmoid后的概率）
        #   target -> [N, H, W] （值为0或1）

        # 展平预测和标签
        #pred = self.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1).float()  # 确保target为浮点型

        # 计算交集和联合
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

class HybridLoss(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.ce = nn.BCEWithLogitsLoss()
            self.dice = DiceLoss()

        def forward(self, pred, target):
            return 0.5*self.ce(pred, target) + self.dice(pred, target)
class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride = 1,padding = 1)
        self.BatchNorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.BatchNorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.result = 0
    def forward(self,in_tensor):
        x = in_tensor
        x = self.conv1(x)
        x = self.BatchNorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        x = self.BatchNorm2(x)
        x = self.relu2(x)
        self.result = x
        return x

class UpSampleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpSampleConv,self).__init__()
        self.upSample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.BatchNorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.BatchNorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()



    def forward(self,x,other_tensor):
        x = self.upSample(x)

        #x = torch.cat([x,other_tensor],dim=1)
        x = self.conv1(x)
        x = self.BatchNorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        x = self.BatchNorm2(x)
        x = self.relu2(x)
        return x

class UpSampleConvCat(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpSampleConvCat,self).__init__()
        self.upSample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(int(in_channels+out_channels), out_channels, kernel_size=3, stride=1, padding=1)
        self.BatchNorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.BatchNorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()



    def forward(self,x,other_tensor):
        x = self.upSample(x)

        x = torch.cat([x,other_tensor],dim=1)
        x = self.conv1(x)
        x = self.BatchNorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        x = self.BatchNorm2(x)
        x = self.relu2(x)
        return x


class UNet(nn.Module):
    def __init__(self, device="cpu"):
        super(UNet,self).__init__()
        DEVICE = device

        self.layer_para = [1,32,64,128,256]
        self.down_layers = nn.ModuleList()
        self.up_layer = nn.ModuleList()
        for i in range(len(self.layer_para)-2):
            if i == 0:
               l = DoubleConv(in_channels=self.layer_para[0], out_channels=self.layer_para[1]).to(DEVICE)

               self.down_layers.append(l)
            else:
                l = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    DoubleConv(in_channels=self.layer_para[i], out_channels=self.layer_para[i+1])).to(DEVICE)
                self.down_layers.append(l)


        #self.down_layers = nn.Sequential(l for _,l in self.down_layers)


        self.last_layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride = 2),
            nn.Conv2d(self.layer_para[-2], self.layer_para[-1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.layer_para[-1]),
            nn.ReLU(),
            nn.Conv2d(self.layer_para[-1], self.layer_para[-1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.layer_para[-1]),
            nn.ReLU())

        for i in range(len(self.layer_para)-1,1,-1):
            self.up_layer.append( UpSampleConvCat( self.layer_para[i], self.layer_para[i-1]).to(DEVICE))
        #self.up_layer.append(UpSampleConv(self.layer_para[3], self.layer_para[2]).to(DEVICE))
        #self.up_layer.append(UpSampleConv(self.layer_para[2], self.layer_para[1]).to(DEVICE))
       # self.up_layers = nn.Sequential(*self.up_layers)

        self.dropout = nn.Dropout(0.5)
        self.map = nn.Conv2d(in_channels= self.layer_para[1],out_channels = 1, kernel_size=3,padding=1)
    def forward(self, x):

        xs = []

        for dc in self.down_layers:
           #print(type(dc))

            x = dc.forward(x)
            xs.append(x)
        x = self.last_layer(x)
        #x = self.dropout(x)
        for uc in self.up_layer:

            x = uc(x,xs.pop())
        x = self.map(x)
        #x = torch.argmax(x,dim=1)
        return x




