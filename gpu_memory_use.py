
import torch

def gpu_memory_use():
  """
  Returns string with amount of gpu memory that is in use.
  If gpu is not available it returns an empty string.
  """

  if torch.cuda.is_available():
    gb = 1024**3
    gb_allocated = torch.cuda.memory_allocated() / gb
    gb_reserved = torch.cuda.memory_reserved() / gb
    return '%.1f / %.1f GB' % (gb_allocated, gb_reserved)

  else:
    return('')