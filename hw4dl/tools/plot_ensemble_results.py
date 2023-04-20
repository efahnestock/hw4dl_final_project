import matplotlib.pyplot as plt 
import torch.nn as nn
from loaders import PolyData
import numpy as np 
from hw4dl.tools.manage_models import load_model, get_most_recent_model
from hw4dl.train import make_polyf
from torch.utils.data import DataLoader

def plot_network_performance(model:nn.Module,
                             toy_loader:PolyData,
                             )->plt.Axes:
    
  fig, ax = plt.subplots()
  ax.set_xlim(toy_loader.lower, toy_loader.upper)
  samples = np.linspace(toy_loader.lower, toy_loader.upper, 1000)
  var = toy_loader.varf(samples)
  ax.plot(samples, toy_loader.polyf(samples), label="True Function")
  ax.fill_between(samples, toy_loader.polyf(samples) - var, toy_loader.polyf(samples) + var, alpha=0.5, label="True Variance")
  ax.scatter(toy_loader.x, toy_loader.y, label="Training Data")
  ax.legend()
  return fig, ax
if __name__ == "__main__":

  most_recent_model = get_most_recent_model()
  model, config = load_model(most_recent_model)
  toy_loader = PolyData(config["polyf"], config["varf"], gaps=config["gaps"])
  polyf, varf, gaps = make_polyf(config["polyf_type"])
  train_dataset = PolyData(polyf, varf, gaps, size=config["train_size"], seed=1111)
  # val_dataset = PolyData(polyf, varf, gaps, size=args.val_size, seed=2222)
  # test_dataset = PolyData(polyf, varf, gaps, size=args.test_size, seed=3333)

  fig, ax = plot_network_performance(model, train_dataset)
  plt.show()