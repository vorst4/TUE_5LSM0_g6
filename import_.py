
import importlib

def import_(name):

  # obtain module <name> from git repo
  mod = importlib.import_module('TUE_5LSM0_g6.' + name)

  # reload module because colab and jupyter notbook are an utter complete piece
  # of garbage that do not reload a module upon change.
  importlib.reload(mod)

  # obtain function with same name as module. This way a module can be loaded in
  # as a function
  func = getattr(mod, name)

  return func
