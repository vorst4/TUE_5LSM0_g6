
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

def make_csv(model, path='/content/ISIC_2019_Test_Input/unk/'):
  """
  first iteration of the function to make the csv file from test images
  """
  print('WARNING: MAKE_CSV IS USED, USE MAKE_CSV2 INSTEAD')
  rows=[]
  test_im_names=[]
  test_imgs=sorted(glob(path+'*.jpg'))
  print(test_imgs)
  for test_im in test_imgs:
    base_name = os.path.basename(test_im)
    base_name=os.path.splitext(base_name)[0]
    test_im_names.append(base_name)

  header=["image","MEL","NV","BCC","AK","BKL","DF","VASC","SCC","UNK"]
  rows.append(header)

  for i in range(len(test_im_names)):
    scores_im=list(map(str,model.test_preds[i]))
    new_row=[test_im_names[i]]+scores_im
    rows.append(new_row)
 
  with open('test_scores.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

