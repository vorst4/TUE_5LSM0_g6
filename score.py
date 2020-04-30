
import torch

class Score():
  def __init__(self, model, N_val, N_classes):

    # check if provided model has the correct type
    if not isinstance(model, torch.nn.Model):
      raise TypeError('model must be of type torch.nn.Model')

    # set attributes
    self._model = model
    self.truth_labels = np.empty((N_val))
    self.val_scores = np.empty(N_val, N_classes)
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # allocate variables



  def calculate():
    pass


  def _evaluate_model():
    """
    Determine performance of the model on the validation set.
    Returns the labels and scores
    """

    # set model to eval (disables dropouts, and prevents calculating the grad)
    self.model.eval()


    # evaluate model on validation set and save the truth_labels & prediction 
    # scores
    i, j = 0, 0
    for x, y in dl_val:
      x = x.to(device=S.device, dtype=torch.float32)
      j += x.shape[0]
      y_true[i:j] = y.numpy()
      scores[i:j, :] = model(x).cpu().detach().numpy()
      i = j

    # return
    return y_true.numpy(), scores.numpy()

def model_performance(S, model_params, dl_val, N_val, N_classes):

  # set it to eval (disables dropouts, and prevents calculating the grad)
  model.eval()


  # evaluate model on validation set
  print('\nEvaluating model on validation set...')
  for x, y in dl_val:
    x = x.to(device=S.device, dtype=S.dtype)
    scores_ = model(x)
    j += x.shape[0]
    y_true[i:j] = y
    scores[i:j, :] = scores_
    i = j
