
import matplotlib.pyplot as plt 
import torch.nn as nn
from hw4dl.loaders.toy_loader import PolyData
import numpy as np 
from hw4dl.tools.manage_models import load_model, get_most_recent_model
from hw4dl.train import make_polyf, make_sigma_positive
from torch.utils.data import DataLoader
from hw4dl.loaders.toy_loader import construct_intervals
import torch
#TODO: check valid interval points, calculate amount of lower,upper that it classified correctly

def get_mask_from_intervals(intervals, x_values:np.ndarray):
  mask = np.zeros_like(x_values, dtype=bool)
  for interval in intervals:
    mask = np.logical_or(mask, np.logical_and(x_values >= interval[0], x_values <= interval[1]))
  return mask

def score_network_performance(model:nn.Module,
                             toy_loader:PolyData,
                             epi_threshold:float=0.1,
                             device:str="cpu",
                             )->plt.Axes:
  samples = np.linspace(toy_loader.lower, toy_loader.upper, 1000)
  var = toy_loader.varf(samples)
  # ax.plot(samples, toy_loader.polyf(samples), label="True Function")

  torch_input = torch.unsqueeze(torch.tensor(samples, dtype=torch.float32), 1).to(device)
  model.to(device)
  model.eval()
  model.scramble_batches = False
  outputs = model(torch_input)
  values = torch.stack(outputs).squeeze(-1)
  means = values[:,:,0].mean(dim=0)
  sigma = torch.sqrt(torch.mean(make_sigma_positive(values[:,:,1]) + torch.square(values[:,:,0]), dim=0) - torch.square(means))
  epistemic_sigma = torch.std(values[:,:,0], dim=0)
  epistemic_sigma = epistemic_sigma.detach().cpu().numpy()

  classified_data_region = epistemic_sigma < epi_threshold
  intervals = construct_intervals(toy_loader.use_gaps, toy_loader.gaps, toy_loader.lower, toy_loader.upper)
  mask = get_mask_from_intervals(intervals, samples)
  samples_correct = np.sum(
    np.logical_and(classified_data_region, mask)
  ) + np.sum(np.logical_and(np.logical_not(classified_data_region), np.logical_not(mask)))
  total_samples = samples.shape[0]

  # positive all uncertainty in no data region minus all uncertainty in data region
  epi_scores = -np.sum(mask * epistemic_sigma) + np.sum(np.logical_not(mask) * epistemic_sigma)


  means = means.detach().cpu().numpy()
  sigma = sigma.detach().cpu().numpy()
  mean_mse = np.mean(np.square(means[mask] - toy_loader.polyf(samples[mask])))
  mean_sigma = np.mean(np.square(np.square(sigma[mask]) - var[mask]))
  print(f"Mean MSE: {mean_mse}")
  print(f"Mean Sigma: {mean_sigma}")
  print(f"Percentage of samples classified correctly: {samples_correct/total_samples}")
  return mean_mse, mean_sigma, samples_correct/total_samples, epi_scores



if __name__ == "__main__":
  most_recent_model = get_most_recent_model()
  model, config = load_model(most_recent_model)
  polyf, varf, gaps = make_polyf(config["polyf_type"])
  train_dataset = PolyData(polyf, varf, gaps, size=config["train_size"], seed=1111)

  score_network_performance(model, train_dataset, epi_threshold=0.01)