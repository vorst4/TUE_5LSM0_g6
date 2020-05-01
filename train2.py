
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

# import backup
import TUE_5LSM0_g6.backup
importlib.reload(TUE_5LSM0_g6.backup)
backup = TUE_5LSM0_g6.backup.backup

# global variables
time_start = 0
epoch = 0
tot_epochs = 0
iteration = 0
prints_per_epoch, cur_print = 0,0
N_classes = 9
N_train = 22799
N_val = 2532
N_test = 8238


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

  global time_start, epoch, tot_epochs, iteration, prints_per_epoch, cur_print

  # add attributes to model, if they do not exist yet
  if not hasattr(model, 'loss'):
    model.loss = []
  if not hasattr(model, 'time_elapsed'):
    model.time_elapsed = []
  if not hasattr(model, 'validation_score'):
    model.validation_score = Score(model)

  # variables
  if not len(model.validation_score.epoch) == 0
    epochs = model.validation_score.epoch[-1] + 1 + np.arange(S.epochs)
    iteration = model.validation_score.iteration[-1]
  else
    epochs = 1 + np.arange(S.epochs)
    iteration = 0
  iter_per_epoch = int( np.ceil( N_train / S.batch_size  ) )
  prints_per_epoch = iter_per_epoch // S.evaluate_every
  tot_epochs = epochs[-1]

  # move model to cpu/gpu
  model = model.to(device=S.device)  # move the model parameters to CPU/GPU

  # start timer
  time_start = time.clock()

  # start (or resume) training
  print('\nEstimated number of iterations per epoch: %i\n' % iter_per_epoch)
  for e in epochs:
    cur_print = 0
    for t, (x, y) in enumerate(dataloader):

      # update current iteration and epoch
      iteration += 1
      epoch = e

      # put model to training mode and move (x,y) to cpu/gpu
      model.train()  
      x = x.to(device=S.device, dtype=S.dtype)
      y = y.to(device=S.device, dtype=torch.long)

      # calculate scores
      scores = model(x)

      # calculate loss(cross etnropy)
      loss = F.cross_entropy(scores, y)
      
      # Zero out all of the gradients for the variables which the optimizer
      # will update.
      optimizer.zero_grad()

      # backward pass, compute loss gradient
      loss.backward()

      # update parameters using gradients
      optimizer.step()
      
      # evaluate model on validation data and print (part of) the results.
      if t % S.evaluate_every == 0:
        cur_print += 1
        _evaluate_and_print(model, time_start)


      # append loss 
      model.loss.append(loss)

    lr_exp.step()
    print(lr_exp.get_lr())


# ---------------------------------------------------------------------------- #
def _evaluate_and_print(model, time_start):

  # evaluate (bma: balanced multiclass accuracy)
  bma, tp, p = model.score.calculate(epoch, iteration)
  

  # print
  var1 = 'Epoch %i/%i' % (epoch, tot_epochs)
  var2 = 'print %i/%i' % (cur_print, prints_per_epoch)
  var3 = 't_elaps.%s' % time_str(t_elap)

  print('Epoch %i/%i, iter %i/%i, t_elaps.%s t_rem.%s, %' % \
        (epoch, tot_epochs, t, N_iter, time_str(t_elap), time_str(t_rem) ) )  

  # backup
  
  str2 = 'loss %.4f %s ' % (loss.item(), accuracy(dl_val, model, S) )
  print(str1+str2)

    model.elapsed_time = time_str(time.clock()-time_start)
    if S.backup_each_epoch:
      backup(model, S.modelname)


def time_str(t):
  return time.strftime('%Hh%Mm%Ss', time.gmtime(t))

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
        nsc[ii] += (y==ii).sum()
    acc = float(num_correct) / num_samples
    if dataloader.dataset.train:
      model.acc_test.append(acc)
    else:
      model.acc_val.append(acc)

    str1 = 'acc %.2f%% (%i/%i)\n' % (100 * acc, num_correct, num_samples)
    str2 = '\t'
    for ii in range(9):
      if ii==4: # add line break
        str2 = str2+'\n\t'
      elif ii==7: # skip class 'unk'
        continue
      str2 = str2 + '%-4s %5i/%-8i ' % (classes[ii], ncc[ii], nsc[ii])
    return str1+str2
  
