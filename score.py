
import torch
import sklearn
import numpy as np
import pandas as pd
from isic_challenge_scoring.classification import ClassificationScore

# ---------------------------------------------------------------------------- #

N_classes = 9
N_val = 2532


# ---------------------------------------------------------------------------- #
class Score():

  # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#
  def __init__(self, model, hyperparam):
    """
    Calculate the performance (score) of the model on the validation set. The 
    provided model must have attribute 'dl_val' containing the validation
    dataloader.

    attributes
    iteration (list, int): list with the iteration at which the score was 
      determined. Note that this must be the total amount of iterations, not 
      the iteration of the current epoch.
    epoch (list, int): epoch
    isic_score (list, object ClassificationScore ): list with objects that are
      created with the isic-challenge-scoring source code.
    balanced_multiclass_accuracy (list, float): the only score that really 
      matters.

    """

    # sanity-check arguments
    if not isinstance(model, torch.nn.Module):
      raise TypeError('model must be of type torch.nn.Module')
    if not hasattr(hyperparam, 'dl_val'):
      raise AttributeError('hyperparam must have attribute dl_val')
    if not isinstance(hyperparam.dl_val, torch.utils.data.DataLoader):
      raise TypeError('dl_val must be of type torch.utils.data.DataLoader')

    # set attributes
    self._model = model
    self._dl_val = hyperparam.dl_val
    self._truth_labels = -np.ones((N_val))
    self._predicted_labels = -np.ones((N_val))
    self._validation_scores = np.zeros((N_val, N_classes))
    self.iteration = []
    self.epoch = []
    self.isic_score = []
    self.balanced_multiclass_accuracy = []


  # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#
  def calculate(self, epoch, iteration):
    """
    Calculate the score of the model using the (crappy) isic-challenge-scoring.

    returns
    TP_class (np array, int, size: N_classes): the number of 
      correctly labeled images per class. This is purely meant for printing
      during training. It is overwritten each time method 'calculate' is called.
    P_class (np array, int, size: N_classes): the number of 
      labels per class of the validation set. This is also meant for printing
      purposes during training and it remains the same.

    Note1: The isic-challenge-scoring will contain nan values if there eixsts a
    label that is not assigned to any image. 
    
    Note2: The training data contains the label 'unk', but none of the 
    training-data-images are assigned to it. This will result in nan values
    (see note1). To prevent this, a random image with label 'nv'  is assigned 
    to 'unk' instead. 'nv' is chosen because it is the biggest class.

    Note3: To prevent any nan values (see note1), labels that aren't assigned
    to any image will be assigned to a single image that contains label 'nv'.
    """

    # sanity-check arguments
    if not isinstance(epoch, int):
      raise TypeError('Epoch must be of type int but %s is given' % type(epoch).__name__)
    if not isinstance(iteration, int):
      raise TypeError('Iteration must be of type int but %s is given' % type(iteration).__name__)
    if not len(self.epoch) == 0:
      if epoch < self.epoch[-1]:
        raise ValueError('epoch %i is given, but last epoch was %i' % (epoch, self.epoch[-1]))
    if not len(self.iteration) == 0:
      if iteration <= self.iteration[-1]:
        raise ValueError('iteration %i is given, but last iteration was %i. Note, iteration is total number of iterations not iteration per epoch' % (iteration, self.iteration[-1]))


    # evaluate model on the validation set
    self._evaluate_model()

    # determine correctly labeled images (TP) and number of labels (P) per class
    TP_class = -np.ones(N_classes)
    P_class  = -np.ones(N_classes)
    for label in range(N_classes):
      pred = self._predicted_labels[label]
      true = self._truth_labels[label]
      TP_class[i] = sum(( label == pred ) & ( label == true ))
      P_class[i]  = sum( label == true )

    # if a label exists that is not assigned to an image (such as 'unk') then 
    # assign it to an image with label 'nv' (biggest class). If no image of 'nv'
    # exists then the function will return and the score will not be calculated.
    # Since it is possible that every image with 'nv' is relabed, sanity-check
    # that all the labels are assigned to an image. If not, simply return and
    # not calculate the score.
    for label in range(N_classes):
      if not np.isin(predicted_labels, label):
        try:
          self._swap_single_label(5, i)
        except:
          return np.Nan, TP_class, P_class
    # sanity-check if every label is assigned to at least one image
    if not len(assigned_labels) == N_classes:
      return np.Nan, TP_class, P_class

    # if the method reaches till here, its possible to determine the 
    # isic-challinge-score
    isic_score = self._calculate_isic_score(self._truth_labels, 
                                            self._validation_scores, 
                                            self._dl_val.datasets.classes)

    # append the score, iteration, epoch, etc. to this class
    self.epoch.append(epoch)
    self.iteration.append(iteration)
    self.isic_score.append(isic_score.to_dict())
    self.balanced_multiclass_accuracy.append(float(isic_score.overall))

    return self.balanced_multiclass_accuracy[-1], TP_class, P_class

  # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#
  def to_dict():
    return {'iteration': self.iteration,
            'epoch': self.epoch,
            'isic_score': self.isic_score,
            'balanced_multiclass_accuracy': self.balanced_multiclass_accuracy}

  # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#
  def from_dict(dic):
    self.iteration = dic['iteration']
    self.epoch = dic['epoch']
    self.isic_score = dic['isic_score']
    self.balanced_multiclass_accuracy = dic['balanced_multiclass_accuracy']

  # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#
  def _evaluate_model(self):
    """
    Determine performance (score) of the model on the validation set and save
    the truth labels.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    self._model.eval()
    i,j = 0,0
    with torch.no_grad():
      for x, y in self._dl_val:
        x = x.to(device=device, dtype=torch.float32)
        j += x.shape[0]
        self._truth_labels[i:j] = y.numpy()
        self._validation_scores[i:j, :] = self._model(x).cpu().detach().numpy()
        i = j
    self._determine_predicted_labels()

  # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#
  def _swap_single_label(self, current_label, new_label):
    """
    Change the first occurence of an image with <current_label> such that its
    label becomes <new_label>. The validation_score that corresponds to that
    image is also changed such that the maximum score corresponds with the new
    label.
    """

    # get image-index of the first occurce of the current_label
    idx_img = (self._truth_labels == 7)[0]

    # change the label the the new_label
    self._truth_labels[idx_img] = new_label

    # also change the validation_score such that it matches the new label
    current_score_idx = current_label
    current_score     = np.max(self._validation_scores[idx_img, current_label])
    max_score         = np.max( self._validation_scores[idx_img, :] )
    max_score_idx     = np.argmax( self._validation_scores[idx_img, :] )
    self._validation_scores[idx_img, current_score_idx] = max_score
    self._validation_scores[idx_img, max_score_idx]     = current_score

    # redetermine the predicted labels
    self._determine_predicted_labels(self)

  # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#
  def _determine_predicted_labels(self):
    """
    returns the predicted labels as indices
    """
    self._predicted_labels = np.argmax(self._validation_scores, axis=1) 

  # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#
  @staticmethod
  def _calculate_isic_score(truth_labels, validation_scores, label_names):
    """
    The isic-challenge-scoring works with pandas.Dataframe. The scoring uses a
    weight with each image, since the weights are unknown to the contestents 
    they are set to 1.0. The format of the truth labels needs to be binary in
    contrast to label-indices, which is what this project works with in general.
    Also, the label-names are used in contrast to label-indices. Furthermore,
    the prediction-scores need to be expressed as a percentage in the following 
    form:
                        1
          ----------------------------
          1  +  exp[- a ( score - b )]

    with the constant a compensating for the mean and constant b setting the 
    threshold such that the sensitivity (recall) is > 89%. 

    TODO: implement a & b such that this is satisfied. They are currently set
    to 1 and 0.5 respectively.
    """

    # define weights (are all set to 1.0)
    ones = np.ones(N_val, type=np.float64)
    weights = pd.DataFrame({'score_weight': ones, 
                            'validation_weight': ones })

    # binary truth labels (shape N_val x N_classes)
    binary_truth_labels = sklearn.preprocessing.LabelBinarizer(truth_labels)

    # binary truth labels as panda.DataFrame
    binary_truth_labels_pd = pd.DataFrame(binary_truth_labels, 
                                          columns=label_names)

    # validation percentage
    a = 1
    b = 0.5
    validation_percentage = 1/(1+np.exp(-a*(validation_scores - b)))
    
    # validation percentage as panda.DataFrame
    validation_percentage_pd = pd.DataFrame(validation_percentage, 
                                            columns=label_names)

    # calculate isic-score
    isic_score = ClassificationScore( binary_truth_labels_pd,
                                      validation_percentage_pd,
                                      weights )

    return isic_score


