
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


def backup(model, 
           modelname,
           model_path='/content/drive/My Drive/5LSM0-final-assignment/' ):
  """
  Function used to backup model, this is replaced by backup2.
  """
  print('WARNING: FUNCTION BACKUP IS USED, USE BACKUP2 INSTEAD !!!')
  date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  path = model_path + modelname + '_' + date_time+'.pt'
  torch.save(model.state_dict(), path)