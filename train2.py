
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
import TUE_5LSM0_g6.backup2
importlib.reload(TUE_5LSM0_g6.backup2)
backup2 = TUE_5LSM0_g6.backup2.backup2

# constants
N_classes = 9
N_train = 22799
N_val = 2532
N_test = 8238


# ---------------------------------------------------------------------------- #

class Train():
  
  def __init__(self, model, hyperparam, model_data, S):
    """
    Trains the specified model and prints the progress.
    It also backups the progress and score if S.backup_each_epoch is set to true.
    
    Args:
      model (torch.nn.Module):  model
      optimizer (torch.optim.Optimizer): optimizer
      epochs (int): number of epochs to train the model

    Returns:
      (none)
    """

    # sanity check
    if not isinstance(model, torch.nn.Module):
      raise TypeError('model must be of type torch.nn.Module')
    if not isinstance(model_data, dict):
      raise InstanceError('model_data needs to be a dictionary')

    # get validation score object
    val_score = hyperparam.score
    self.hyperparam = hyperparam

    # add keys to dictionary if its empty. If not, restore val-score object
    if not model_data:
      model_data['loss'] = []
      model_data['validation_score'] = []
    else:
      val_score.from_dict(model_data['validation_score'])


    # set class attributes
    if not len(val_score.epoch) == 0:
      epochs = val_score.epoch[-1] + 1 + np.arange(S.epochs)
      self.iteration = val_score.iteration[-1]
    else:
      epochs = 1 + np.arange(S.epochs)
      self.iteration = 0
    self.iter_per_epoch = int( np.ceil( N_train / S.batch_size  ) )
    self.prints_per_epoch = int( np.ceil( self.iter_per_epoch / S.evaluate_every ) )
    self.epoch_end = epochs[-1]
    self.iteration_start = self.iteration
    self.iteration_end = self.epoch_end * self.iter_per_epoch

    # move model to cpu/gpu
    model = model.to(device=S.device)  # move the model parameters to CPU/GPU

    # start timer
    self.time_start = time.clock()

    # start (or resume) training
    print('\nEstimated number of iterations per epoch: %i\n' % self.iter_per_epoch)
    for e in epochs:
      self.cur_print = 0
      for t, (x, y) in enumerate(hyperparam.dl_train):

        # update current iteration and epoch
        self.iteration += 1
        self.epoch = e

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
        hyperparam.optimizer.zero_grad()

        # backward pass, compute loss gradient
        loss.backward()

        # update parameters using gradients
        hyperparam.optimizer.step()

        # append loss 
        model_data['loss'] = np.append( model_data['loss'], loss.item() )
        
        # evaluate model on validation data and print (part of) the results.
        if t % S.evaluate_every == 0:
          self.cur_print += 1
          self._evaluate_and_print(model, val_score, loss)

      # bakcup model (if required)
      if S.backup_each_epoch:
        model_data['validation_score'] = val_score.to_dict()
        backup2(model, model_data, S.modelname)

      # update learning rate
      hyperparam.lr_exp.step()
      print('\n new lr = ', hyperparam.lr_exp.get_lr())


  # ---------------------------------------------------------------------------- #
  def _evaluate_and_print(self, model, score, loss):

    # evaluate (bma: balanced multiclass accuracy)
    bma, tp, p = score.calculate(int(self.epoch), int(self.iteration))
    t_elap = time.clock() - self.time_start
    t_per_iter = t_elap / (self.iteration - self.iteration_start)
    t_rem = (self.iteration_end - self.iteration) * t_per_iter

    # print
    #   line 1
    var1 = 'Epoch %i/%i  ' % (self.epoch, self.epoch_end)
    var2 = 'print %i/%i  ' % (self.cur_print, self.prints_per_epoch)
    var3 = 't_elaps.%s  t_rem.%s  ' %  (time_str(t_elap), time_str(t_rem))
    var4 = 'loss %.4f  ' % loss.item()
    var5 = 'bma %.3f  ' % bma
    line1 = var1 + var2 + var3 + var4 + var5 + '\n'
    #   line 2 and 3
    line23 = '    '
    label_names = self.hyperparam.dl_train.dataset.classes
    for label in range(N_classes):
      if label==4: # add line break
        line23 += '\n    '
      elif label==7: # skip class 'unk'
        continue
      line23 += '%-4s %5i/%-8i ' % (label_names[label] , tp[label], p[label] )
    #   print them
    print( line1 + line23 )

# ---------------------------------------------------------------------------- #
def time_str(t):
  return time.strftime('%Hh%Mm%Ss', time.gmtime(t))

# ---------------------------------------------------------------------------- #


