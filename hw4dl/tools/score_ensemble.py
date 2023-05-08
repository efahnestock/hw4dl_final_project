
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
  """
  Get a mask of the x_values that are in the intervals
  :param intervals: The intervals to check
  :param x_values: The x values to check
  :return: A mask of the x values that are in the intervals 
  """
  mask = np.zeros_like(x_values, dtype=bool)
  for interval in intervals:
    mask = np.logical_or(mask, np.logical_and(x_values >= interval[0], x_values <= interval[1]))
  return mask

def score_network_performance(model:nn.Module,
                             toy_loader:PolyData,
                             epi_threshold:float=0.1,
                             device:str="cpu",
                             )->tuple[np.ndarray, np.ndarray, float, float]: 
  """
  Score ensemble performance on a toy dataset!
  :param model: The model to score
  :param toy_loader: The toy dataset
  :param epi_threshold: The threshold to use for epistemic uncertainty. Deprecated
  :param device: The device to run the model on
  :return: A tuple of the mean MSE, mean sigma, percentage of samples classified correctly, and epistemic score
  """
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
  sigma = make_sigma_positive(values[:,:,1]).mean(dim=0)
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

def get_mask_from_gaps(gaps, shape):
  mask = np.ones(shape)
  for gap in gaps:
    xx, yy = np.meshgrid(list(range(gap[0], gap[1] + 1)), list(range(gap[2], gap[3] + 1)))
    mask[xx, yy] = 0
  return mask.flatten()

def score_cnn_performance(model:nn.Module,
                             test_loader,
                             epi_threshold,
                             device,
                             )->plt.Axes:
  model.eval()
  testx, testy = np.meshgrid(np.arange(15), np.arange(15))
  coords = []
  all_inputs = []
  for x, y in zip(testx.ravel(), testy.ravel()):
    inputs = np.zeros((15, 15))
    inputs[x, y] = 1
    all_inputs.append(torch.tensor(inputs))
    coords.append((x, y))
  coords = np.array(coords)
  inputs = torch.stack(all_inputs).unsqueeze(1).type(torch.float32).to(device)
  outputs = model(inputs)
  values = torch.stack(outputs).squeeze(-1)
  means = values[:, :, 0].mean(dim=0)
  sigma = torch.sqrt(
    torch.mean(make_sigma_positive(values[:, :, 1]) + torch.square(values[:, :, 0]), dim=0) - torch.square(means))
  epistemic_sigma = torch.std(values[:, :, 0], dim=0)
  epistemic_sigma = epistemic_sigma.detach().cpu().numpy()
  classified_data_region = epistemic_sigma < epi_threshold

  mask = get_mask_from_gaps(test_loader.gaps, test_loader.shape).astype(bool)

  x_input = (2 * testx.ravel()) / test_loader.shape[0] - 1
  y_input = (2 * testy.ravel()) / test_loader.shape[0] - 1
  gt_mean = test_loader.polyf(x_input, y_input)
  gt_var = test_loader.varf(x_input, y_input)

  samples_correct = np.sum(
    np.logical_and(classified_data_region, mask)
  ) + np.sum(np.logical_and(np.logical_not(classified_data_region), np.logical_not(mask)))
  total_samples = len(testx) * len(testy)

  # positive all uncertainty in no data region minus all uncertainty in data region
  epi_scores = -np.sum(mask * epistemic_sigma) + np.sum(np.logical_not(mask) * epistemic_sigma)

  means_arr = means.detach().cpu().numpy()
  sigma_arr = sigma.detach().cpu().numpy()
  mean_mse = np.mean(np.square(means_arr[mask] - gt_mean[mask]))
  mean_sigma = np.mean(np.square(np.square(sigma_arr[mask]) - gt_var[mask]))
  print(f"Mean MSE: {mean_mse}")
  print(f"Mean Sigma: {mean_sigma}")
  print(f"Percentage of samples classified correctly: {samples_correct / total_samples}")
  print(f"Epi score {epi_scores}")
  return mean_mse, mean_sigma, samples_correct/total_samples, epi_scores

if __name__ == "__main__":
  most_recent_model = get_most_recent_model()
  model, config = load_model(most_recent_model)
  polyf, varf, gaps = make_polyf(config["polyf_type"])
  train_dataset = PolyData(polyf, varf, gaps, size=config["train_size"], seed=1111)

  score_network_performance(model, train_dataset, epi_threshold=0.01)