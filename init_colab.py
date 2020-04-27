import os
import json
import torch
from google.colab import drive


def init_colab():
  """
  Initializes colab by 
    1.  Linking Google Drive 
    2.  Extracting train and test data from drive.
    3.  Cloning and setting up GIT repo
    4.  Removing the default 'sample_data' folder since this is not used during 
        the project. 
  
  Note: If the  test data folder already exists in the current workfolder, the 
  function assumes colab is already initialized and it skips this.

  Args:
    (None)

  Returns:
    (None)
  """

  # Do nothing if test folder already exists
  if os.path.exists('ISIC_2019_Test_Input'):
    return

  # mount google drive
  print('\nMounting Google Drive...')
  drive.mount('/content/drive')
  print('...Done')

  # Extract train and test data from drive
  #   copy data.zip from Google Drive to workfolder .
  print('\nCopying data.zip to workfolder...')
  os.system("cp 'drive/My Drive/5LSM0-final-assignment/data.zip' .")
  print('...Done')
  #   unpack data.zip
  print('\nUnpacking data.zip...')
  os.system('unzip -q data.zip')
  print('...Done')
  #   remove data.zip
  print('\nRemoving data.zip...')
  os.system('rm data.zip')
  print('...Done')

  # setup git
  print('\nSetting up git...')
  #   load github.json
  with open('/content/drive/My Drive/github.json', 'r') as json_file:
    gitconfig = json.load(json_file)
  #   link <username> and <key> to repo
  url = 'https://'+gitconfig["username"]+':'+gitconfig["key"]+'@github.com/vorst4/TUE_5LSM0_g6.git'
  os.system('git -C TUE_5LSM0_g6 remote set-url origin ' + url)
  #   move to git directory and set <username> and <email>
  os.system('git -C TUE_5LSM0_g6 config user.name '+gitconfig["username"])
  os.system('git -C TUE_5LSM0_g6 config user.email '+gitconfig["email"])
  print('...Done')

  # remove default sample_data folder, since it is unrelated to this project.
  print('\nRemoving sample_data...')
  os.system('rm -r sample_data')
  print('...Done')

  # check if GPU is enabled
  if torch.cuda.is_available() == False:
    print('\nWARNING: GPU is not enabled !!!')
