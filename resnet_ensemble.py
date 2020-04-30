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
  def __init__(self, block, num_classes=5):
    super(ResNet, self).__init__()
    self.backup_restore_name_prefix = 'resnet18_'
    self.backup_restore_path = '/content/drive/My Drive/5LSM0-final-assignment/'
    self.loss = []

    self.in_planes = 64

    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=strides[0], padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 64, stride=strides[1])
    self.layer2 = self._make_layer(block, 128, stride=strides[2])
    self.layer3 = self._make_layer(block, 256, stride=strides[3])
    self.layer4 = self._make_layer(block, 512, stride=strides[4])
    self.linear = nn.Linear(512*block.expansion, num_classes)

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

  def print_layer_sizes(self):
    empty_np = np.zeros((1, 3, img_size, img_size), dtype=np.float32)
    empty_tensor = torch.from_numpy(empty_np)
    print('\nBelow are the output sizes of each layer (input img: %ix%i)' %
          (img_size, img_size))
    # print(empty_tensor.shape)
    out = F.relu(self.bn1(self.conv1(empty_tensor)))
    print('  conv1  ', out.shape)
    out = self.layer1(out)
    print('  layer1 ', out.shape)
    out = self.layer2(out)
    print('  layer2 ', out.shape)
    out = self.layer3(out)
    print('  layer3 ', out.shape)
    out = self.layer4(out)
    print('  layer4 ', out.shape)
    out = F.avg_pool2d(out, 4)
    print('  pool   ', out.shape)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    print('  dense  ', out.shape)

  def backup_to_drive(self):
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = self.backup_restore_path + self.backup_restore_name_prefix + date_time+'.pt'
    torch.save(self.state_dict(), path)

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
    plt.plot(x, self.loss)
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.show()


# ---------------------------------------------------------------------------- #

strides = []
img_size = 0

def resnet18(image_size):
  global strides, img_size

  # set img_size
  img_size = image_size

  # chose strides for a given input size such that the dense layer has size 1x1
  if img_size == 32:
    strides = [1, 1, 2, 2, 2]
  elif img_size == 64:
    strides = [1, 2, 2, 2, 2]
  elif img_size == 128:
    strides = [2, 2, 2, 2, 2]
  elif img_size == 256:
    strides = [3, 2, 2, 2, 2]
  elif img_size == 512:
    strides = [3, 3, 3, 2, 2]
  elif img_size == 1024:
    strides = [3, 3, 3, 3, 2]
  else:
    raise ValueError('ERROR: invalid image size')

  return ResNet(BasicBlock)