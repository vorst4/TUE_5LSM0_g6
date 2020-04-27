
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
from glob import glob
from datetime import datetime
from PIL import Image


# ---------------------------------------------------------------------------- #

def train(model, optimizer, dataloader, dl_val, lr_exp, S):
  """
  Trains the specified model and prints the progress
  
  Args:
    model (torch.nn.Module):  model
    optimizer (torch.optim.Optimizer): optimizer
    epochs (int): number of epochs to train the model

  Returns:
    (none)
  """
  model = model.to(device=S.device)  # move the model parameters to CPU/GPU
  model.loss = []
  model.acc_val = []
  model.acc_test = []
  model.elapsed_time = []
  time_start = time.clock()
      
  for e in range(S.epochs):
    for t, (x, y) in enumerate(dataloader):
      model.train()  # put model to training mode
      x = x.to(device=S.device, dtype=S.dtype)  # move to device, e.g. GPU
      y = y.to(device=S.device, dtype=torch.long)
      scores = model(x)
      loss = F.cross_entropy(scores, y)
      
      # Zero out all of the gradients for the variables which the optimizer
      # will update.
      optimizer.zero_grad()

      # backward pass, comput loss gradient
      loss.backward()

      # update parameters using gradients
      optimizer.step()
      
      # append loss 
      model.loss.append(loss)

      # update plot
      if t % S.print_every == 0:
        time_elapsed = time.strftime('%H:%M:%S', time.gmtime(time.clock()-time_start))
        stri = accuracy(dl_val, model, S)
        print('Iteration %d, loss = %.4f, time = %s, %s' % 
              (t, loss.item(), time_elapsed, stri))
        

    lr_exp.step()
    model.elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.clock()-time_start))
    if S.backup_each_epoch:
      model.backup_to_drive()

    print(lr_exp.get_lr())


# ---------------------------------------------------------------------------- #

def accuracy(dataloader, model, S):
  """
  Calculate accuracy of given model
  """
  num_correct = 0
  num_samples = 0
  ncc = np.zeros(9)
  nsc = np.zeros(9)
  classes = dataloader.dataset.classes
  model.eval()  # set model to evaluation mode
  with torch.no_grad():
    for x, y in dataloader:
      x = x.to(device=S.device, dtype=S.dtype)  # move to device, e.g. GPU
      y = y.to(device=S.device, dtype=torch.long)
      scores = model(x)
      _, preds = scores.max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
      for ii in range(9):
        ncc[ii] += ( (preds == ii) & (y==ii)).sum()
        nsc[ii] += (preds==ii).sum()
    acc = float(num_correct) / num_samples
    if dataloader.dataset.train:
      model.acc_test.append(acc)
    else:
      model.acc_val.append(acc)

    str1 = 'acc = %.2f, %d/%d correct\n' % (100 * acc, num_correct, num_samples)
    str2 = '\t'
    for ii in range(9):
      if ii==4: # add line break
        str2 = str2+'\n\t'
      elif ii==7: # skip class 'unk'
        continue
      str2 = str2 + '%-4s %3i/%-5i ' % (classes[ii], ncc[ii], nsc[ii])
    return str1+str2
  
