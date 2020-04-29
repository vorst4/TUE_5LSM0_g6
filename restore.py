
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
  
  # construct path, if no date_time is given get most recent model
  if date_time == '':
    models = glob(model_path + modelname + '*.pt')
    if len(models) == 0:
      print('\nERROR: No models found with given modelname\n')
      return model
    path = models[-1]
  else:
    path = model_path+modelname+'_'+date_time
    if not os.path.exists(path):
      print('\nERROR: Path of provided modelname and date_time not found\n')
      return model

  # restore the model
  if torch.cuda.is_available():
    model.load_state_dict(torch.load(path))
    model.eval()
    print('\nRestored model\n')
  else:
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
  print('\nWARNING: Using CPU !!!\nRestored model\n')

  return model
  
