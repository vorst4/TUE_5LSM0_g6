
import os
import time
import json
import shutil
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


def download_and_format(mnt_path='/content/drive',
                        token_path='/',
                        data_path='/5LSM0-final-assignment',
                        desired_size = 512):
  """
  This function donwloads and formats the data. More specifically:
  1.  Google Drive is mounted to colab.
  2.  Everything in the current folder that is related to the files that are to 
      be downloaded, or to the files that this function creates are DELETED.
      This is a precaution to prevent file conflicts.
  3.  Download the ISIC 2019 data from Kaggle. Your Kaggle token is needed for 
      this. To download your Kaggle token, visit https://www.kaggle.com , then 
      go to 'my account' and then 'download API token'. To pass your Kaggle
      token to the script, Google Drive is used. This way, multiple people can
      run/share the same script without sharing sensitive data. The token 
      should have the name 'kaggle.json' and be placed in Google Drive at the
      path specified by 'token_path'.
  3.  The downloaded data is unziped.
  4.  The width and height of the images are resized to 'desired_size' and 
      zero padding is added if necessary.
  4.  The train data/images are organized into subfolders. Each image is placed 
      into a subfolder that corresponds to its class. Doing this makes it
      possible to use 'torchvision.datasets.ImageFolder'.
  6.  A zip file is created that contains the reformatted and reorganized test
      and training data. The reason for this is because when saving the data and
      loading it from Google Drive at a later time, a time-out error can occur.
      A Google Drive timeout can occur when a folder contains lots (thousands) 
      of files, which is currently the case.
  7.  The zip file is moved to Google Drive, to a folder with name 
      '5LSM0-final-assignment'. Make sure that enough space is available, since 
      the function does not check this!
  5.  All the files, except data.zip on Google Drive, are deleted.

  Todo:  link the metadata to the image, this is currently ignored.
  
  Args:
    mnt_path (string): path where the google drive will be mounted.
    token_path (string): Path on Google Drive where the token can be found.
      With '/' being the root of Google Drive.
    data_path (string): Path on Google Drive where data.zip will be stored. 
      With '/' being the root of Google Drive.
    desired_size (int): desired image width and height.

  Returns:
    (none)
  """

  # make sure google drive is mounted
  print('\nMounting Google Drive...')
  drive.mount(mnt_path, force_remount=True)
  print('...Done')

  # remove files (in case this is a redownload)
  print('\nDeleting old files...')
  !rm -r ISIC_2019_Test_Input
  !rm -r ISIC_2019_Training_Input
  !rm -r isic-2019-training-input.zip
  !rm -r isic-2019-training-groundtruth.zip
  !rm -r isic-2019-test-input.zip
  !rm -r isic-2019-training-metadata.zip
  !rm -r isic-2019-test-metadata.zip
  !rm -r ISIC_2019_Training_GroundTruth.csv
  !rm -r ISIC_2019_Training_Metadata.csv
  !rm -r ISIC_2019_Test_Metadata.csv
  print('...Done')

  # Set kaggle configuration directory
  os.environ['KAGGLE_CONFIG_DIR'] = mnt_path+'/My Drive'+token_path

  # download data
  print('\nDownloading...')
  !kaggle datasets download -d kioriaanthony/isic-2019-training-input
  !kaggle datasets download -d kioriaanthony/isic-2019-training-groundtruth
  !kaggle datasets download -d kioriaanthony/isic-2019-test-input
  # !kaggle datasets download -d kioriaanthony/isic-2019-training-metadata
  # !kaggle datasets download -d kioriaanthony/isic-2019-test-metadata
  print('...Done')

  # unzip it (quietly)
  print('\nUnzipping...')
  !unzip -q isic-2019-training-input.zip
  !unzip -q isic-2019-training-groundtruth.zip
  !unzip -q isic-2019-test-input
  # !unzip -q isic-2019-test-metadata.zip
  # !unzip -q isic-2019-training-metadata.zip
  print('...Done')

  # resize and zero-padd images to desired size
  print_every = 1000
  print('\nResizing and padding images...')
  paths = glob('/content/ISIC_2019_Test_Input/*.jpg') + \
          glob('/content/ISIC_2019_Training_Input/*.jpg')
  N_files = len(paths)
  t_start = time.clock()
  N_cur = 0
  for path in paths:
    # print progress 
    N_cur = N_cur+1
    if N_cur % print_every  == 0:
      N_remain = N_files - N_cur
      t_elapse = time.clock()-t_start
      t_rem = N_remain * t_elapse/N_cur
      t_elapse_str = time.strftime('%Mm %Ss', time.gmtime(t_elapse))
      t_rem_str = time.strftime('%Mm %Ss', time.gmtime(t_rem))
      print('Img %i / %i , elapsed time %s, estimated remaining time %s' % (N_cur, N_files, t_elapse_str, t_rem_str))
    # load image
    img = Image.open(path)
    # resize
    resized_size = np.array([img.width, img.height])*desired_size//max(img.size)
    img = img.resize(resized_size)
    # pad
    empty_img = Image.new("RGB", (desired_size, desired_size))
    paste_location = tuple((desired_size - resized_size)//2)
    empty_img.paste(img, paste_location)
    img = empty_img
    # save img
    img.save(path)
  print('...Done')

  # create subfolders with class name (if they do not exist yet) 
  print('\nCreating subdirs...')
  # path to training and test folder
  paths = ['/content/ISIC_2019_Training_Input/',
           '/content/ISIC_2019_Test_Input/']
  classes = ['mel', 'nv', 'bcc', 'ak', 'bkl', 'df', 'vasc', 'scc', 'unk']
  for path in paths:
    print('creating subdirs in: '+path)
    for clas in classes:
      if not os.path.exists(path+clas):
        os.mkdir(path+clas)
        print('created dir: '+clas)
      else:
        print('dir: '+clas+' already exists')
  print('...Done')

  # obtain the class that corresponds to each img and move it to its folder.
  print('\nReading classing and moving images')
  # Training data
  with open('ISIC_2019_Training_GroundTruth.csv', 'r') as f:
    next(f) # skip header (first line) of .csv file
    for line in f:
      # obtain image name and corresponding class name
      arr = np.array(line.split(','))
      img = arr[0]+'.jpg' # img name
      idx = np.where( arr[1:].astype(np.float) == 1.0 )[0][0] # class-index
      clas = classes[idx]
      # move image (if it exists)
      if os.path.exists(paths[0]+img):
        os.rename(paths[0]+img, paths[0]+clas+'/'+img) # rename = move
      else :
        print('img not found: '+img)
  # Test data (every class is unknown, so move everything to unk)
  for img in glob(paths[1]+'*.jpg'):
    name = img.split('/')[-1]
    os.rename(img, paths[1]+'unk/'+name)
  print('...Done')

  # # add test and training data to zip (quietly and recursive)
  print('\nCreating data.zip containing training and test data...')
  !zip -q -r data.zip ISIC_2019_Test_Input ISIC_2019_Training_Input
  print('...Done')

  # move zip to drive
  print('\nMoving data to Google Drive...')
  shutil.move('/content/data.zip', mnt_path+'/My Drive'+data_path+'/data.zip')
  print('...Done')

  # remove left over files
  print('\nDeleting left-over files...')
  !rm -r isic-2019-training-input.zip
  !rm -r isic-2019-training-groundtruth.zip
  !rm -r isic-2019-test-input.zip
  !rm -r isic-2019-training-metadata.zip
  !rm -r isic-2019-test-metadata.zip
  !rm -r ISIC_2019_Training_GroundTruth.csv
  !rm -r ISIC_2019_Training_Metadata.csv
  !rm -r ISIC_2019_Test_Metadata.csv
  !rm -r ISIC_2019_Test_Input
  !rm -r ISIC_2019_Training_Input
  print('...Done')

download_and_format()