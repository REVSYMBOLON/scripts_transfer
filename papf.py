import torch
import torch.nn as nn
from collections import OrderedDict



def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

class YoloBody(nn.Module):
    def __init__(self, feature_map1, feature_map2, feature_map3):
        super(YoloBody, self).__init__()
    

        self.feature_map1 = feature_map1
        self.feature_map2 = feature_map2
        self.feature_map3 = feature_map3


        self.down_sample1       = conv2d(128,256,3,stride=2)
        self.down_sample2       = conv2d(256,512,3,stride=2)

        self.conv_for_P4        = conv2d(512,256,1)


    def forward(self, x):

        f1_down = self.down_sample1(self.feature_map1)
        f1_weight = torch.sigmoid(f1_down)
        out2 = torch.cat([f1_weight.mul(self.feature_map2),f1_down],axis=1)

        f2_down = self.down_sample1(self.feature_map1)
        f2_weight = torch.sigmoid(f2_down)
        out3 = torch.cat([f2_weight.mul(self.feature_map3),f2_down],axis=1)

        

        return self.feature_map1, out2, out3
