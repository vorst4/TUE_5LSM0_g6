
import numpy as np
import cv2
from google.colab.patches import cv2_imshow



# ---------------------------------------------------------------------------- #

def create_heatmaps(model, tensor_in, idx):
  """
  Args:
  model (torch.nn.Module): The pytorch model
  tensor_in (torch.Tensor): input tensor to the model, it should contain 
    <batch_size> amount of images.
  idx (int): the index of the image in <tensor_in> of which the heatmap should
    be created.
  """

  # obtain image size from tensor
  img_size = tensor_in.shape[2]

  # original image, size (img_size, img_size, 3)
  img = tensor_in[idx].numpy().swapaxes(0,-1).astype(np.uint8)

  # Obtain the last convolutional layer (before the dense layer).
  #   has size (batch_size, conv_planes, conv_kernel_size, conv_kernel_size)
  last_features = model.extract_features(tensor_in)

  # obtain feature <idx> of the batch.
  #   has size (conv_planes, conv_kernel_size, conv_kernel_size)
  last_feature = last_features[idx].detach().numpy()

  # Obtain the weights of the dense layer after the last conv layer
  #   Size (conv_planes, N_classes)
  weight_softmax = np.squeeze(list(model.parameters())[-2].data.numpy())

  # multiple the two to obtain cam (class activation map)
  #   Size (N_classes, conv_kernel_size, conv_kernel_size)
  cam = np.dot( weight_softmax, last_feature.swapaxes(0, 1))

  # transform cam such that it lies within the range of [0, 255] (greyscale)
  tmp = cam - np.min(cam)
  cam_tran = (255 * tmp / np.max(tmp)).astype(np.uint8)

  # obtain a heatmap per label
  heatmaps = np.empty((N_classes, img_size, img_size, 3), dtype=np.uint8)
  for label in range(N_classes):

    # upscale it to (img_size, img_size)
    grey_img = cv2.resize(cam_tran[label],(img_size, img_size)) 

    # convert it to a 'heatmap' (img_size, img_size)
    heatmaps[label, :, :, :] = cv2.applyColorMap(grey_img, cv2.COLORMAP_JET)

  # return input image and heatmap
  return img, heatmaps



# ---------------------------------------------------------------------------- #

def combine_heatmaps_and_image(img, heatmaps):
  # concatenate the original image with its heatmaps
  img_size = img.shape[0]
  img_combi = np.zeros((img_size, (N_classes+1)*img_size, 3))
  for label in range(N_classes+1):
    indices = range(label*img_size, (label+1)*img_size)
    if label == 0:
      img_combi[:, indices, :] = img
    else:
      img_combi[:, indices, :] = heatmaps[label-1]
  return img_combi


# ---------------------------------------------------------------------------- #

def display_random_heatmap(model, dl_train):

  # obtain random tensor_in with corresponding truth labels
  tensor_in, truth_labels = _get_batch_train_imgs(dl_train)

  # generate heatmaps
  idx = 0
  img, heatmaps = create_heatmaps(model, tensor_in, idx)

  # combine them into a single image
  img_combi = combine_heatmaps_and_image(img, heatmaps)

  # print info and display image
  print('Generated heatmap of random image. heatmap %i is the correct one' % truth_labels[0])
  cv2_imshow(img_combi)


# ---------------------------------------------------------------------------- #

def _get_batch_train_imgs(dl_train):
  for t, (x, y) in enumerate(dl_train):
    return x.detach(), y.detach().numpy()




