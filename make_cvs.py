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


def make_csv(model, path='/content/ISIC_2019_Test_Input/unk/'):
  rows=[]
  test_im_names=[]
  test_imgs=sorted(glob(path+'*.jpg'))
  for test_im in test_imgs:
    base_name = os.path.basename(test_im)
    base_name=os.path.splitext(base_name)[0]
    test_im_names.append(base_name)

  header=["image","MEL","NV","BCC","AK","BKL","DF","VASC","SCC","UNK"]

  rows.append(header)
  for i in range(len(test_im_names)):
    scores_im=list(map(str,model.test_preds[i]))
    score_sorted = []
    score_sorted.append(scores_im[4])
    score_sorted.append(scores_im[5])
    score_sorted.append(scores_im[1])
    score_sorted.append(scores_im[0])
    score_sorted.append(scores_im[2])
    score_sorted.append(scores_im[3])
    score_sorted.append(scores_im[8])
    score_sorted.append(scores_im[6])
    score_sorted.append(scores_im[7])
    new_row=[test_im_names[i]]+score_sorted
    rows.append(new_row)

  with open('test_scores.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

  
