
import torch
import pandas as pd
from isic_challenge_scoring.classification import ClassificationScore


# ---------------------------------------------------------------------------- #

N_classes = 9
N_val = 2532


# ---------------------------------------------------------------------------- #
class Score():

  # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#
  def __init__(self, model, dl_val):
    """
    Init
    """

    # sanity-check the type of each argument
    if not isinstance(model, torch.nn.Model):
      raise TypeError('model must be of type torch.nn.Model')
    if not isinstance(dl_val, torch.utils.data.DataLoader):
      raise TypeError('dl_val must be of type torch.utils.data.DataLoader')

    # set attributes
    self._model = model
    self._dl_val = dl_val
    self._truth_labels = np.empty((N_val))
    self._validation_scores = np.empty(N_val, N_classes)
    self._weights = pd.DataFrame()
    self._weights.score_weight = np.ones(N_val)
    self._weights.validation_weight = np.ones(N_val)
    self.iteration = []


  # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#
  def calculate(self, iteration):
    """
    Calculate the score of the model using the (crappy) isic-challenge-scoring.

    Note1: The isic-challenge-scoring will contain nan values if there eixsts a
    label that is not assigned to any image. 
    
    Note2: The training data contains the label 'unk', but none of the 
    training-data-images are assigned to it. This will result in nan values
    (see note1). To prevent this, a random image with label 'nv'  is assigned 
    to 'unk' instead. 'nv' is chosen because it is the biggest class.

    Note3: To prevent any nan values (see note1), labels that aren't assigned
    to any image will be assigned to a single image that contains label 'nv'.
    """

    # sanity-check type of argument
    if not type(iteration) is int:
      raise TypeError('Iteration must be of type int')

    # evaluate model on the validation set
    self._evaluate_model()

    # if a label exists that is not assigned to an image (such as 'unk') then 
    # assign it to an image with label 'nv' (biggest class). If no image of 'nv'
    # exists then the function will return and the score will not be calculated.
    predicted_labels = self._predicted_labels()
    for label in range(N_classes):
      if not np.isin(predicted_labels, label):
        try:
          self._swap_single_label(5, i)
        except:
          return

    # check if label 0-8 is assigned to at least one image. Do not calculate
    # the score and return if this is not the case. It's possible that 'nv' is
    # not assigned to any image
    assigned_labels = np.unique(self._predicted_labels())
    if not len(assigned_labels) == N_classes:
      return

    # If the function made it here, it means that the isic-challenge-scoring
    # can be calculated without nan values or errors. This method uses a weight
    # for each image, since these weights are unknown to the contestors, a
    # weight value of 1.0 is used for each image.

  # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#
  def _evaluate_model(self):
    """
    Determine performance (score) of the model on the validation set and save
    the truth labels.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    i,j = 0,0
    for x, y in self._dl_val:
      x = x.to(device=device, dtype=torch.float32)
      j += x.shape[0]
      self._truth_labels[i:j] = y.numpy()
      self._validation_scores[i:j, :] = model(x).cpu().detach().numpy()
      i = j

  # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#
  def _swap_single_label(self, current_label, new_label):
    """
    Change the first occurence of an image with <current_label> such that its
    label becomes <new_label>. The validation_score that corresponds to that
    image is also changed such that the maximum score corresponds with the new
    label.
    """

    # get image-index of the first occurce of the current_label
    idx_img = self._truth_labels == 7)[0]

    # change the label the the new_label
    self._truth_labels[idx_img] = new_label

    # also change the validation_score such that it matches the new label
    current_score_idx = current_label
    current_score     = np.max(self._validation_scores[idx_img, current_label])
    max_score         = np.max( self._validation_scores[idx_img, :] )
    max_score_idx     = np.argmax( self._validation_scores[idx_img, :] )

    self._validation_scores[idx_img, current_score_idx] = max_score
    self._validation_scores[idx_img, max_score_idx]     = current_score

  # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#
  def _predicted_labels(self):
    return np.argmax(self._validation_scores, axis=1)


