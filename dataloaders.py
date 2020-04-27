

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


def dataloaders(root='/content/',
                train_img_dir='ISIC_2019_Training_Input/',
                test_img_dir='ISIC_2019_Test_Input/',
                batch_size=64,
                val_ratio = 5):
  """
  This function creates and returns dataloaders for the train, validation and 
  test data set.

  Args:
    root (str): root directory
    train_img_dir (str): directory containing images of the training set
    test_img_dir (str): directory that contains the images of the test set
    batch_size (int): batch size
    validation_size_percentage (float, 0...100): the train data is split into train 
      and valuation. This percentage specifies how much of the original data is 
      used as validation. The remaining percentage will be the new train data.
  
  Returns: 
    train (torch.utils.data.DataLoader): object to load training data
    val (torch.utils.data.DataLoader): object to load validation data
    test (torch.utils.data.DataLoader): object to load test data
    N_train (int): number of training images
    N_val (int): number of validation images
    N_test (int): number of test images
  """

  # dataset transformation. Expand the dataset by adding random horzontal and 
  # vertical flips. 
  # todo: add more data transformations (but not to the test set ofcourse)
  # todo: the training image shapes are either 1024x1024 or 600x450. This is
  #       for the moment hotfixed by resizing it to 32x32. Note that, when 
  #       changing this, the neural net also needs to be changed
  # todo: data normalization
  transform_train = T.Compose([T.RandomHorizontalFlip(), 
                               T.RandomVerticalFlip(),
                               T.Resize((128,128)),  # todo: temporary fix
                               T.ToTensor()])
  transform_test  = T.ToTensor()

  # datasets
  dataset_train = dset.ImageFolder(root+train_img_dir, transform=transform_train)
  dataset_test = dset.ImageFolder(root+test_img_dir, transform=transform_test)

  # split train into validation and (new) train
  N = len(dataset_train.imgs)
  N_train = int(np.round(N*(100-val_ratio)/100))
  N_val = N-N_train
  N_test = len(dataset_test.imgs)

  # indices related to validation and the (new) training set.
  random_indices = np.array(range(N))
  np.random.shuffle(random_indices)
  indices_train = random_indices[:N_train]
  indices_val  =random_indices[N_train:]

  # samplers
  sampler_train = sampler.SubsetRandomSampler(indices_train)
  sampler_val = sampler.SubsetRandomSampler(indices_val)

  # dataloaders
  dl_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train)
  dl_val = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_val)
  dl_test = DataLoader(dataset_test, batch_size=batch_size)

  # add bool to see if certain dataset is the training dataset
  dl_train.dataset.train = True
  dl_val.dataset.train = True
  dl_test.dataset.train = False

  return dl_train, dl_val, dl_test
