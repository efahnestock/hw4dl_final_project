import torch, json, datetime 
import torch.nn as nn
import os 
from hw4dl.models.separated_network import ToyNet
from hw4dl.models.shared_backbone import VariableBackbone
from hw4dl import ROOT_DIR

def save_model(model:nn.Module, extra_metrics:dict, args):
  # save model with current date as filename 
  weights_filename =os.path.join(ROOT_DIR, f"weights/{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H:%M:%S')}.pt")
  config_filename = os.path.join(ROOT_DIR, f"weights/{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H:%M:%S')}.json")

  save_metrics = extra_metrics
  save_metrics.update(vars(args))
  torch.save(model, weights_filename)
  with open(config_filename, "w") as f:
    f.write(json.dumps(save_metrics, sort_keys=True, indent=2, separators=(',', ': ')))
    
def load_model(base_filepath:str):
  json_path = base_filepath + ".json"
  pt_path = base_filepath + ".pt"
  with open(json_path, "r") as f:
    config = json.load(f)

  model = torch.load(pt_path)
  return model, config
    
def get_most_recent_model():
  """
  Get the most recent base path of a saved model
  """
  files = os.listdir(os.path.join(ROOT_DIR, "weights"))
  files = [f for f in files if f.endswith(".json")]
  files.sort()
  return os.path.join(ROOT_DIR, "weights/", files[-1][:-5])