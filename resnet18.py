
import os
import time
import json
import shutil
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from google.colab import drive
from glob import glob
from datetime import datetime
from PIL import Image


# ---------------------------------------------------------------------------- #

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ---------------------------------------------------------------------------- #

class ResNet(nn.Module):
  def __init__(self, block, num_classes=9):
    super(ResNet, self).__init__()
    self.backup_restore_name_prefix = 'resnet18_'
    self.backup_restore_path = '/content/drive/My Drive/5LSM0-final-assignment/'
    self.loss = []

    self.in_planes = 64

    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 64, stride=1)
    self.layer2 = self._make_layer(block, 128, stride=2)
    self.layer3 = self._make_layer(block, 256, stride=2)
    self.layer4 = self._make_layer(block, 512, stride=2)
    self.linear = nn.Linear(512*block.expansion*4, num_classes)

  def _make_layer(self, block, planes, stride):
    strides = [stride,1] 
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out

  def backup_to_drive(self):
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = self.backup_restore_path + self.backup_restore_name_prefix + date_time+'.pt'
    torch.save(model.state_dict(), path)

  def restore_from_drive(self, path):
    self.load_state_dict(torch.load(path))
    self.eval()

  def restore_latest(self):
    models = glob(self.backup_restore_path + self.backup_restore_name_prefix + '*.pt')
    try:
      self.restore_from_drive(models[-1])
    except:
      print('Error during restoring latest backup')
  
  def visualize(self):
    fig=plt.figure(figsize=(12, 6), dpi= 80, facecolor='w', edgecolor='k')
    x = np.array(range(len(self.loss)))
    plt.plot(x, model.loss)
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.show()


# ---------------------------------------------------------------------------- #

def resnet18():
    return ResNet(BasicBlock)

