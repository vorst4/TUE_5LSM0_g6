
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


def restore(model, 
            modelname, 
            model_path='/content/drive/My Drive/5LSM0-final-assignment/',
            date_time=''):
  
  # if no date_time is given, use the latest
  if date_time == '':
    models = glob(model_path + modelname + '*.pt')
    path = models[-1]
  else:
    path = model_path+modelname+'_'+date_time

  # restore the model using the path
  model.load_state_dict(torch.load(path))
  model.eval()

  return model
  
