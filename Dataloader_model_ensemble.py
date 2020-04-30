

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

import normalize_data
importlib.reload(normalize_data)
normalize_data = normalize_data.normalize_data


def dataloaders(root='/content/',
                train_img_dir='ISIC_2019_Training_Input/',
                test_img_dir='ISIC_2019_Test_Input/',
                batch_size=64,
                img_size=64):
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
  transform_train = T.Compose([#T.RandomHorizontalFlip(), 
                               #T.RandomVerticalFlip(),
                               T.Resize((img_size,img_size)),  # todo: temporary fix
                               T.ToTensor(),
                               T.Normalize((0.00036133, 0.00027880,  0.00027078),
                                           (0.01646914, 0.01290044, 0.012730150))
                               ])
  transform_val =  T.Compose([T.Resize((img_size,img_size)),
                               T.ToTensor(),
                               T.Normalize((0.00036133, 0.00027880,  0.00027078),
                                           (0.01646914, 0.01290044, 0.012730150))
                               ])

  transform_test  = T.Compose([T.Resize((img_size,img_size)),
                               T.ToTensor(),
                               T.Normalize((0.00036133, 0.00027880,  0.00027078),
                                           (0.01646914, 0.01290044, 0.012730150))
                               ])

  # datasets
  dataset_train = dset.ImageFolder(root+train_img_dir, transform=transform_train)
  dataset_val = dset.ImageFolder(root+train_img_dir, transform=transform_val)
  dataset_test = dset.ImageFolder(root+test_img_dir, transform=transform_test)

  # split the (old) training data into (new) training data (90%) and test 
  # data (10%). The split is done per class, the first 90% is taken for the
  # (new) training data en de last 10% for the validation data.
  val_ratio = 0.1
  N_classes = 9

  datasplit1=[0,3,6,7,8]
  for i in range(5):
    # indices of class <i>
    idxs1 = np.where(np.array(dataset_train.targets) == datasplit1[i])[0]
    # determine the number of (new) training samples
    N_train1 = int(np.round(len(idxs1)*(1-val_ratio)))
    # split the (old) training set into (new) training and validation set
    if i==0:
      idxs_train1 = idxs1[:N_train1]
      idxs_val1 = idxs1[N_train1:]
    else:
      idxs_train1 = np.concatenate([idxs_train1, idxs1[:N_train1]])
      idxs_val1 = np.concatenate([idxs_val1, idxs1[N_train1:]])


  datasplit2=[1,2,4,5]
  for i2 in range(4):
   # indices of class <i>
   idxs2 = np.where(np.array(dataset_train.targets) == datasplit2[i2])[0]
   # determine the number of (new) training samples
   N_train2 = int(np.round(len(idxs2)*(1-val_ratio)))
   # split the (old) training set into (new) training and validation set
   if i2==0:
     idxs_train2 = idxs2[:N_train2]
     idxs_val2 = idxs2[N_train1:]
   else:
     idxs_train2 = np.concatenate([idxs_train2, idxs2[:N_train2]])
     idxs_val2 = np.concatenate([idxs_val2, idxs2[N_train2:]])

  # samplers
  sampler_train1 = sampler.SubsetRandomSampler(idxs_train1)
  sampler_train2 = sampler.SubsetRandomSampler(idxs_train2)
  sampler_val1 = sampler.SubsetRandomSampler(idxs_val1)
  sampler_val2 = sampler.SubsetRandomSampler(idxs_val2)
  # print(len(idxs_train))
  # print(len(idxs_val))

  # dataloaders
  dl_train1 = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train1)
  dl_train2 = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train2)
  
  dl_val1 = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val1)
  dl_val2 = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val2)

  dl_test = DataLoader(dataset_test, batch_size=batch_size)

  # normalize_data(dl_train)

  # add bool to see if certain dataset is the training dataset
  dl_train1.dataset.train = True
  dl_val1.dataset.train = False
  dl_train2.dataset.train = True
  dl_val2.dataset.train = False
  dl_test.dataset.train = False

  return dl_train1, dl_train2, dl_val1,dl_val2, dl_test