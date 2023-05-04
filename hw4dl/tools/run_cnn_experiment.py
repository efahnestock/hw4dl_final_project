from collections import namedtuple
import os, torch
import numpy as np
import datetime
import sys
sys.path.append("/data/vision/phillipi/perception/hw4dl_final_project")
sys.path.append("/data/vision/phillipi/perception/hw4dl_final_project/hw4dl")
from hw4dl import ROOT_DIR
from hw4dl.main import parse_options
from hw4dl.train_cnn import train
from hw4dl.loaders.map2loc_loader import Map2Loc
from hw4dl.tools.manage_models import load_model
from hw4dl.tools.score_ensemble import score_cnn_performance
from hw4dl.tools.plot_ensemble_results import plot_cnn_performance
import json
import pandas as pd

ExpConfig = namedtuple("exp_config",
                        ["name",
                         "split_indexes",
                         "task",
                         "seed",
                         "device",
                        ])

def set_all_seeds(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)

def run_experiment(exp_config:ExpConfig, args):
  exp_name = exp_config.name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  base_exp_path = os.path.join(ROOT_DIR, "experiments", exp_name)
  # create experiment directory
  os.makedirs(base_exp_path)
  # save experiment config
  with open(os.path.join(base_exp_path, "config.json"), "w") as f:
    json.dump(exp_config._asdict(), f)

  results = dict(split_idx=[], mean_mse=[], sigma_mse=[], per_correct=[])

  for split_idx in exp_config.split_indexes:
    set_all_seeds(exp_config.seed)
    print("here2")
    # train network
    args = argparse.Namespace(
    device_type="cuda",
    batch_size=16,
    model_type='single',
    split_idx=split_idx,
    num_heads=3,
    lr=1e-3,
    n_epochs=1,
    task=exp_config.task,
    scramble_batches=False,
    )
    args.split_idx = split_idx
    args.device_type =  exp_config.device
    split_idx_dir= os.path.join(base_exp_path, f"split_{split_idx}")
    os.makedirs(split_idx_dir)

    train(args=args, device=torch.device(args.device_type), save_path=split_idx_dir)

    model_filename = None
    for file in os.listdir(split_idx_dir):
      if file.endswith(".pt"):
        model_filename = os.path.basename(file)[:-3]
    model, config = load_model(os.path.join(split_idx_dir, model_filename))

    # evaluate network performance
    test_dataset = Map2Loc(root_dir=f'/data/vision/phillipi/perception/hw4dl_final_project/hw4dl/datasets/map2loc_{args.task}_test', csv_file='description.csv')
    mean_mse, sigma_mse, per_correct = score_cnn_performance(model, test_dataset, 0.01)
    results["split_idx"].append(split_idx)
    results["mean_mse"].append(mean_mse)
    results["sigma_mse"].append(sigma_mse)
    results["per_correct"].append(per_correct)
    # print out performance

    # create plot
    plot_cnn_performance(model, test_dataset, base_exp_path, device=args.device_type)
    # fig.savefig(os.path.join(base_exp_path, f"{split_idx:03d}_performance.png"))

    # save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(base_exp_path, "results.csv"), index=False)

if __name__ == "__main__":
  print("here")
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--name", type=str, default="fc_experiment")
  parser.add_argument("--split_indexes", type=int, nargs="+", default=[0, 1, 2, 3])
  parser.add_argument("--seed", type=int, default=1111)
  parser.add_argument("--task", type=str, default="pixel")
  parser.add_argument("--device_type", type=str, default="cpu")
  args = parser.parse_args()
  exp_config = ExpConfig(name=args.name, split_indexes=args.split_indexes, seed=args.seed, task=args.task,device=args.device_type)
  run_experiment(exp_config, exp_config)
