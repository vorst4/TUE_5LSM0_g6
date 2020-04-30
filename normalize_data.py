
import numpy as np
import torch



def normalize_data(dl_train):
  """
  Determines the mean and standard deviation of the provided list of datasets.
  The mean and std can be used to normalize these datasets.
  Note: This function does not normalize the datasets, its determines the 
  parameters to do so!

  args
  datasets (1D list of datasets objects): the datasets

  output:
    mean (tuple, 3 elements): mean per color (RGB)
    std (tuple, 3 elements): standerd deviation per color (RGB)
  """

  dataset_train = dl_train.dataset

  # for att in dir(dl_test):
  #   print(att)

  mean = torch.zeros(3)
  std = torch.zeros(3)
  n_imgs = 0

  print('---------')
  
  # add all the imgs up and count the number of imgs
  for t, (x, y) in enumerate(dl_train):
    n_imgs += x.shape[0]
    try:
      x_sum += x
      x_sum2 += x ** 2
    except:
      x_sum = x
      x_sum2 = x ** 2

    print('n_imgs = ', n_imgs)

  # total number of pixels (n_imgs * width * height)
  n_pixels = n_imgs * x_sum.shape[2] * x_sum.shape[3]

  # calculate mean and std for each rgb
  mean = x_sum.sum([0, 2, 3]) / n_pixels
  std = np.sqrt( x_sum2.sum([0, 2, 3])/n_pixels - mean**2 )

  # convert tensor to numpy array
  mean = mean.numpy()
  std = std.numpy()
  
  print('mean = ', mean)
  print('std = ', std)
