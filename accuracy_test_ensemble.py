
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


def get_accuracy_test_ensemble(loader1,loader2, model1,model2,S):

  model1.eval()  # set model to evaluation mode
  model1.test_preds=[]
  model2.eval()
  model2.test_preds=[]
  total_score = []
  with torch.no_grad():
    for x1, y1 in loader1:
      x1 = x1.to(device=S.device, dtype=S.dtype)  # move to device, e.g. GPU
      y1 = y1.to(device=S.device, dtype=torch.long)
      scores1 = model1(x1)
      _, preds1 = scores1.max(1)
      s1=score1.tolist()
      model1.test_preds.append(s1)
 
    for x2, y2 in loader2:
      x2 = x2.to(device=S.device, dtype=S.dtype)  # move to device, e.g. GPU
      y2 = y2.to(device=S.device, dtype=torch.long)
      scores2 = model2(x2)
      _, preds2 = scores2.max(1)
      s2=score2.tolist()
      model2.test_preds.append(s2)
    for i in range(len(model1.test_preds)):
        temp_list =[]
        temp_list.append(model1.test_preds[i][0])
        temp_list.append(model2.test_preds[i][0])
        temp_list.append(model2.test_preds[i][1])
        temp_list.append(model1.test_preds[i][1])
        temp_list.append(model2.test_preds[i][2])
        temp_list.append(model2.test_preds[i][3])
        temp_list.append(model1.test_preds[i][2])
        temp_list.append(model1.test_preds[i][3])
        temp_list.append(model1.test_preds[i][4])
        total_score.append(temp_list)
        
        
        
        