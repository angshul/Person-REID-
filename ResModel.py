import torch
import torch.nn as nn
import torch.nn.functional as F

#from model import resnet50
from model import resnet50


class Model(nn.Module):
  def __init__(self, last_conv_stride=2):
    super(Model, self).__init__()
    self.model = resnet50(pretrained=True, last_conv_stride=last_conv_stride)
    self.globalpool = nn.AdaptiveAvgPool2d((1,1))

  def forward(self, x):
    # shape [N, C, H, W]
    x = self.model(x)
    #x = F.avg_pool2d(x, x.size()[2:])
    #shape [N,C,1,1]
    x = self.globalpool(x)
    # shape [N, C]
    x = x.view(x.size(0), -1)

    return x
