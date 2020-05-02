import os
import time
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
import random
import shutil
import csv
import math


class EnsembleModel(nn.Module):
  def __init__(self, model1, model2, out_size=9):
    super().__init__()
    self.model1 = model1
    self.model2 = model2
    self.model1.linear = nn.Identity()
    self.model2.fc_layer = nn.Identity()
    self.classifier = nn.Linear(2048+342, out_size)
  
  def forward(self,x):
    x1 = self.model1(x.clone()) 
    x1 = x1.view(x1.size(0), -1)
    x2 = self.model2(x)
    x2 = x2.view(x2.size(0), -1)
    out= torch.cat((x1, x2), dim=1)
    out =self.classifier(F.relu(out))
    return out

