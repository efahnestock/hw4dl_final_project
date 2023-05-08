from collections import namedtuple
import os, torch
import numpy as np 
import datetime 
from hw4dl import ROOT_DIR
from hw4dl.main import parse_options
from hw4dl.train import train, make_polyf
from hw4dl.loaders.toy_loader import PolyData
from hw4dl.tools.manage_models import load_model
from hw4dl.tools.score_ensemble import score_network_performance
from hw4dl.tools.plot_ensemble_results import plot_network_performance
import json
import pandas as pd 

ExpConfig = namedtuple("exp_config", 
                        ["name",
                         "split_indexes",
                         "seed", 
                         "device",
                        ])

def set_all_seeds(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)

def run_experiment(exp_config:ExpConfig):
  exp_name = exp_config.name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  base_exp_path = os.path.join(ROOT_DIR, "experiments", exp_name)
  # create experiment directory
  os.makedirs(base_exp_path)
  # save experiment config
  with open(os.path.join(base_exp_path, "config.json"), "w") as f:
    json.dump(exp_config._asdict(), f)

  results = dict(split_idx=[], mean_mse=[], sigma_mse=[], per_correct=[], epi_score=[])

  set_all_seeds(exp_config.seed)
  for split_idx in exp_config.split_indexes:
    # train network
    args = parse_options()
    args.split_idx = split_idx
    args.device_type =  exp_config.device
    args.scrambe_batches = True
    split_idx_dir= os.path.join(base_exp_path, f"split_{split_idx}")
    os.makedirs(split_idx_dir)

    train(args=args, device=torch.device(args.device_type), save_path=split_idx_dir)

    model_filename = None 
    for file in os.listdir(split_idx_dir):
      if file.endswith(".pt"):
        model_filename = os.path.basename(file)[:-3]
    model, config = load_model(os.path.join(split_idx_dir, model_filename))
    # evaluate network performance
    polyf, varf, gaps = make_polyf(config["polyf_type"])
    train_dataset = PolyData(polyf, varf, gaps, size=config["train_size"], seed=1111)
    mean_mse, sigma_mse, per_correct, epi_score = score_network_performance(model, train_dataset, 0.01)
    results["split_idx"].append(split_idx)
    results["mean_mse"].append(mean_mse)
    results["sigma_mse"].append(sigma_mse)
    results["per_correct"].append(per_correct)
    results["epi_score"].append(epi_score)
    # print out performance 

    # create plot
    fig, ax = plot_network_performance(model, train_dataset)
    fig.savefig(os.path.join(base_exp_path, f"{split_idx:03d}_performance.png"))

    # save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(base_exp_path, "results.csv"), index=False)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--name", type=str, default="fc_experiment")
  parser.add_argument("--split_indexes", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
  parser.add_argument("--seed", type=int, default=1111)
  parser.add_argument("--device_type", type=str, default="cuda")
  args = parser.parse_args()
  exp_config = ExpConfig(name=args.name, split_indexes=args.split_indexes, seed=args.seed, device=args.device_type)
  run_experiment(exp_config)
