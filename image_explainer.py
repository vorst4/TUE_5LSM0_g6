
import shap
import numpy as np


# ---------------------------------------------------------------------------- #

def image_explainer():

  batch = next(iter(dl_val))
  imgs, _ = batch
  img = imgs[:1] # grab only 1 img from batch
  tmp = imgs[:1]
  print(tmp.size())

  e = shap.DeepExplainer(model, img)
  shap_values = e.shap_values(img)

  shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
  test_numpy = np.swapaxes(np.swapaxes(img.numpy(), 1, -1), 1, 2)

  shap.image_plot(shap_numpy, -test_numpy)
  return()
