
import numpy as np
import torch
from datetime import datetime
from glob import glob


N_classes = 9
header=["image","MEL","NV","BCC","AK","BKL","DF","VASC","SCC","UNK"]


# ---------------------------------------------------------------------------- #
def make_csv(model, dl_test, modelname, test_scores=None):
  """
  This function is used to generate test_scores and a csv file that can be
  uploaded to the 2019 ISIC challenge.
  """

  # get scores of the model on the test set
  if test_scores is None:
    print('no test scores provided, calculating them now...')
    test_scores = _evaluate_test(model, dl_test)

  # convert scores to probabilities as required by ISIC
  a = 1
  b =0.5
  test_probabilities = 1/ (1 + np.exp(-a*(test_scores-b)))

  # check if every class is present at least once
  predicted_labels = np.argmax( test_probabilities, axis=1 )
  labels_present = np.unique(predicted_labels)
  number_labels_present = len(labels_present)
  if not number_labels_present == N_classes:
    print('WARNING: Not all labels are present, ', labels_present)

  # if not, swap te labels of an image from NV class (biggest class) with the 
  # label that is not present.
  for label in range(N_classes):
    if np.sum( label == labels_present ) == 0:
      print('swapping label %i with %i' % (5, label))
      test_probabilities = _swap_single_label(5, label, predicted_labels, test_probabilities)

  # double check if all labels are present
  predicted_labels = np.argmax( test_probabilities, axis=1 )
  labels_present = np.unique(predicted_labels)
  number_labels_present = len(labels_present)
  if not number_labels_present == N_classes:
    raise ValueError('ERROR: STILL NOT ALL LABELS ARE PRESENT ', labels_present)

  # get labels from class
  labels = dl_test.dataset.class_to_idx
  print(labels)

  # image names per index
  samples = np.array(dl_test.dataset.samples)
  img_paths = samples[:, 0]
  img_names = np.empty(N_test, dtype="S13")
  for i in range(N_test):
    splt = img_paths[i].split('/')
    name = splt[-1].split('.')[0]
    img_names[i] = name
  img_names = img_names.astype('str')

  # indices of sorted img_names
  indices = np.argsort( img_names )
  
  # create output string
  #   header
  output = ''
  for col in header:
    output += col + ','
  output = output[:-1] + '\n'

  #   data
  for i in indices:
    output += img_names[i] + ','
    output += ( '%.16f' % test_probabilities[i, 4]) + ','
    output += ( '%.16f' % test_probabilities[i, 5]) + ','
    output += ( '%.16f' % test_probabilities[i, 1]) + ','
    output += ( '%.16f' % test_probabilities[i, 0]) + ','
    output += ( '%.16f' % test_probabilities[i, 2]) + ','
    output += ( '%.16f' % test_probabilities[i, 3]) + ','
    output += ( '%.16f' % test_probabilities[i, 8]) + ','
    output += ( '%.16f' % test_probabilities[i, 6]) + ','
    output += ( '%.16f' % test_probabilities[i, 7]) + '\n'

  # path of csv file
  date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  path = '/content/drive/My Drive/5LSM0-final-assignment/csv/' + modelname + '_test_scores_'+date_time+'.csv'
  print(path)
  
  # write csv file
  with open(path, "w") as text_file:
    text_file.write(output)

  # done
  print('sucessfully generated .csv file')


# ---------------------------------------------------------------------------- #
def _evaluate_test(model, dl_test):
  """
  Determine performance (score) of the model on the test set
  """
  N_test = len(dl_test.dataset)
  test_scores = np.zeros((N_test, N_classes))
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model.eval()
  k = 0
  with torch.no_grad():
    for x, y in dl_test:
      # check if x is indeed at index k
      if not np.sum( x.numpy() - dl_test.dataset.__getitem__(k)[0].numpy() ) == 0:
        print('ERROR: not the same at index', k)
      x = x.to(device=device, dtype=torch.float32)
      test_scores[k, :] = model(x).cpu().detach().numpy()
      if k % 100 == 0:
        print('evaluating sample %i / %i' % (k , N_test))
      k+=1
  return test_scores
  


# ---------------------------------------------------------------------------- #
def _swap_single_label(current_label, new_label, predicted_labels, test_probabilities):
  """
  Change the first occurence of an image with <current_label> such that its
  label becomes <new_label>. The validation_score that corresponds to that
  image is also changed such that the maximum score corresponds with the new
  label.
  """

  # get image-index of the first occurce of the current_label
  idx_img = np.where( predicted_labels == current_label )[0][0]

  # swap validation scores
  score_cur = test_probabilities[idx_img, current_label]
  score_new = test_probabilities[idx_img, new_label]
  test_probabilities[idx_img, new_label]     = score_cur
  test_probabilities[idx_img, current_label] = score_new

  # return probabilities
  return test_probabilities




