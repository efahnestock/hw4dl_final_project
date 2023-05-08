import sys
# sys.path.append("/data/vision/phillipi/perception/hw4dl_final_project")
# sys.path.append("/data/vision/phillipi/perception/hw4dl_final_project/hw4dl")
# sys.path.append("/data/vision/phillipi/perception/hw4dl_final_project/hw4dl/datasets")
from hw4dl.loaders.map2loc_loader import Map2Loc
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from hw4dl.models.separated_network import ToyNet
from hw4dl.models.shared_cnn import VariableCNNBackbone
import numpy as np
import datetime, json
import pdb
from hw4dl.tools.manage_models import save_model
from torch.distributions.normal import Normal

TOY_LAYER_SHAPES = {
    "pixel": (1, 16, -1, 32, -1, 64, 128, 'fc512', 'fc2'),
    "patch": (1, 16, -1, 32, -1, 64, 128, 256, 'fc512', 'fc18')
}
PIXEL_DATASET_PATH = 'datasets/map2loc_pixel'
PATCH_DATASET_PATH = 'datasets/map2loc_patch'

def make_sigma_positive(sigma):
  return torch.log(1 + torch.exp(sigma)) + 1e-6

def nll_loss(outputs, labels):
  """
  Negative log likelihood loss
  Outputs: B x 2
  Labels: B x 1
  """
  mu, sigma = torch.split(outputs, 1, dim=1)
  sigma = make_sigma_positive(sigma)
  cond_dist_x = Normal(loc=mu, scale=sigma)
  loss = -cond_dist_x.log_prob(labels)

  return loss.mean()

def eval(model, loader, criterion, device):
    """
    Evaluates the input model on the given test set.
    :param model: A model of type VariableCNNBackbone
    :param loader: A dataloader of type Map2Loc.
    :param criterion: A loss object to calculate model loss
    :param device: "cuda" or "cpu"
    :return: Test loss calculated with the input criterion function.
    """
    model.eval()
    total_loss = 0.0
    for inputs, labels in tqdm(loader):
        batch_loss = 0.0
        inputs, labels = inputs.type(torch.float32).to(device), labels.type(torch.float32).to(device)
        outputs = model(inputs)
        for head_i, output in enumerate(outputs):
            batch_loss += criterion(output, labels)

        total_loss += batch_loss.item()
    return total_loss / len(loader)

def train(args, device, save_path=None):
    """
    Trains a model (defined by the args argument).
    :param args: A Namespace object, defined in run_cnn_experiment.py
    :param device: "cuda" or "cpu"
    :param save_path: Path to save checkpoints and model training results.
    :return: None
    """
    if args.task == "pixel":
        train_dataset = Map2Loc(root_dir=PIXEL_DATASET_PATH, csv_file='description.csv')
        test_dataset = Map2Loc(root_dir=PIXEL_DATASET_PATH + "_test", csv_file='description.csv')
    else:
        train_dataset = Map2Loc(root_dir=PATCH_DATASET_PATH, csv_file='description.csv')
        test_dataset = Map2Loc(root_dir=PATCH_DATASET_PATH + "_test", csv_file='description.csv')

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.8),
                                                                               int(len(train_dataset) * 0.2)])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = VariableCNNBackbone(TOY_LAYER_SHAPES[args.task], args.split_idx, args.num_heads, input_size=(15, 15), task=args.task)
    model.to(device)

    criterion = nll_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    val_loss = eval(model, args.model_type, args.scramble_batches, val_loader, criterion, device)
    print("Initial val loss, ", val_loss)
    # Training loop
    for i in range(args.n_epochs):
        model.train()
        total_train_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            batch_loss = 0.0
            inputs = inputs.type(torch.float32).to(device)
            labels = labels.type(torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            for head_i, output in enumerate(outputs):
                batch_loss = criterion(output, labels)

            batch_loss.backward()
            optimizer.step()
            total_train_loss += batch_loss.item()

        train_loss = total_train_loss / len(train_loader)
        val_loss = eval(model, args.model_type, args.scramble_batches, val_loader, criterion, device)
        print(f"Epoch {i}, train loss: {train_loss}, val loss: {val_loss}")

    test_loss = eval(model, args.model_type, args.scramble_batches, test_loader, criterion, device)
    print(f"Test loss: {test_loss}")

    print(f"Saving model and config")
    save_model(model, dict(test_loss=test_loss), args, save_dir=save_path)

    print("Done :)")



