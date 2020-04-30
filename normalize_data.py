
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
  n_samples = 0
  x_sum = torch.zeros(3)

  for t, (x, y) in enumerate(dl_train):
    x = x.to(device=S.device, dtype=S.dtype)  # move to device, e.g. GPU
    batch, _, h, w = x.shape
    n_samples += batch
    n_pixels = batch*h*w

    if t % 100 == 0:
      print('t=')
      print(t)
    

    x_sum += x.sum([0, 2, 3])

  print(t)
  print(n_samples)

  asdf()

  #   asdf()
  #   mean += x.mean

  #   tmp1 = torch.sum(data, dim=[0, 2, 3])
  #   fst_moment = (cnt * mean + sum_) / (cnt + nb_pixels)
  #   # tmp2 = 
  #   print(x.shape)
  #   print(val.shape)


  #   asdf()


  #   mean = 0.
  #   std = 0.
  #   nb_samples = 0.
  #   for data in loader:
  #       batch_samples = data.size(0)
  #       data = data.view(batch_samples, data.size(1), -1)
  #       mean += data.mean(2).sum(0)
  #       std += data.std(2).sum(0)
  #       nb_samples += batch_samples

  #   mean /= nb_samples
  #   std /= nb_samples


  #   for data in data_loader:

  #       b, c, h, w = data.shape
  #       nb_pixels = b * h * w
  #       sum_ = torch.sum(data, dim=[0, 2, 3])
  #       sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
  #       fst_moment = (cnt * mean + sum_) / (cnt + nb_pixels)
  #       snd_moment = (cnt * std + sum_of_square) / (cnt + nb_pixels)

  #       cnt += nb_pixels

  #   return mean, torch.sqrt(std - mean ** 2)


  # print(dl_test.sampler)

  # print(len(dl_test.targets))

  # print(dl_test.dataset)
  # mean = np.mean(dataset_train)
  # std = np.std(dataset_train)

  # print(mean)
  # print(std)

  asdf()