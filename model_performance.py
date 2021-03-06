
import torch
import numpy as np
from src.efficientnet.efficientnet_pytorch.model import EfficientNet

from restore import restore

def model_performance(S, model_params, dl_val, N_val, N_classes):
  """
  Determines performance of provided model on the validation set.
  Returns the labels and scores
  """
  print('WARNING: MODEL_PERFORMANCE IS USED, THIS FUNCTION SHOULD HAVE BECOME OBSOLETE')

  # # free gpu memory
  # if torch.cuda.is_available():
  #   torch.cuda.empty_cache()

  # create model
  model = EfficientNet.from_name(S.modelname, model_params)

  # restore latest model from drive
  model = restore(model, S.modelname)

  # push model to gpu
  model = model.to(device=S.device)

  # set it to eval (disables dropouts, and prevents calculating the grad)
  model.eval()

  # pre allocate variables
  y_true = np.empty(N_val)
  scores = np.empty((N_val, N_classes))
  i,j = 0,0

  # evaluate model on validation set
  print('\nEvaluating model on validation set...')
  for x, y in dl_val:
    x = x.to(device=S.device, dtype=S.dtype)
    j += x.shape[0]
    y_true[i:j] = y.numpy()
    scores[i:j, :] =   model(x).cpu().detach().numpy()
    i = j

  # return
  return y_true, scores

