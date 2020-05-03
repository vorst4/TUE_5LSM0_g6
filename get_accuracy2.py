

def get_accuracy(loader, model):
  """
  Function calculates accuracy but is not used any more.
  """
  print('WARNING: GET_ACCURACY IS USED, USE SCORE INSTEAD')
  
  num_correct = 0
  num_samples = 0
  model.eval()  # set model to evaluation mode
  model.test_preds=[]
  #if (model.__class__.__name__) == "EnsembleModel":
  #  model.acc_val=[]
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
      y = y.to(device=device, dtype=torch.long)
      scores = model(x)
      _, preds = scores.max(1)
      if loader == dl_test:
        for score in scores:
          s=score.tolist()
          model.test_preds.append(s)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    if loader==dl_val or loader==dl_train:
      model.acc_val.append(acc)
      print('acc = %.2f, %d/%d correct' % (100 * acc, num_correct, num_samples))
    else:
      model.acc_test.append(acc)
    