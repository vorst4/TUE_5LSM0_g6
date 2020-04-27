
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

def train(model, optimizer, dataloader, dl_val, S):
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
        

    my_lr_scheduler.step()
    model.elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.clock()-time_start))
    if backup_after_epoch:
      model.backup_to_drive()

    print(my_lr_scheduler.get_lr())


# ---------------------------------------------------------------------------- #

def accuracy(loader, model, S):
  """
  Calculate accuracy of given model
  """
  num_correct = 0
  num_samples = 0
  model.eval()  # set model to evaluation mode
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device=S.device, dtype=S.dtype)  # move to device, e.g. GPU
      y = y.to(device=S.device, dtype=torch.long)
      scores = model(x)
      _, preds = scores.max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    if loader.dataset.train:
      model.acc_test.append(acc)
    else:
      model.acc_val.append(acc)
    return 'acc = %.2f, %d/%d correct' % (100 * acc, num_correct, num_samples)
  
