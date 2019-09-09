from __future__ import print_function, division

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, features, bias=False, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(features)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out += x
        out = self.relu(out)
        out = self.bn2(out)
        return out

class MyModel(nn.Module):
    def __init__(self, img_channels=3, out_features=64, num_res_blocks=5, bias=True, dtype=torch.FloatTensor):
        super(MyModel, self).__init__()
        layers = []
        stride = 1
        
        layers.append(nn.Conv2d(img_channels, out_features, kernel_size=3, padding=1, stride=stride, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(out_features))
        for i in range(num_res_blocks):
            layers.append(ResidualBlock(out_features, bias=bias, stride=stride))
        layers.append(nn.Conv2d(out_features, 32, kernel_size=1, padding=0, stride=stride, bias=bias))
        layers.append(nn.Conv2d(32, 16, kernel_size=1, padding=0, stride=stride, bias=bias))
        layers.append(nn.Conv2d(16, img_channels, kernel_size=1, padding=0, stride=stride, bias=bias))
        
        self.model = nn.Sequential(*layers).type(dtype)
   
    def forward(self, x):
        output = self.model(x)
        output = x - output
        return output