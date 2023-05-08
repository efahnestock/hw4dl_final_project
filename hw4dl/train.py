from hw4dl.loaders.toy_loader import PolyData
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from hw4dl.models.separated_network import ToyNet
from hw4dl.models.shared_backbone import VariableBackbone
import numpy as np
import datetime, json
import pdb
from hw4dl.tools.manage_models import save_model
from torch.distributions.normal import Normal
layer_width = 30
TOY_LAYER_SHAPES = [1] + [layer_width] * 5 + [2]

def make_polyf(typex):
    """
    Return a polynomial function, and associated variance function and gaps for constructing a dataset
    """
    if typex == "cubic":
        def polyf(x):
            return 2*x ** 3 + 5*x ** 2 + 2

        def varf(x):
            return (1-x)**2 * np.log(1+np.exp(x))

        gaps = [(-0.5, 0.5)]
        # gaps = [(-1.0, -0.4), (0.5, 0.7), (0.1, 0.3)]
        return polyf, varf, gaps
    else:
        raise ValueError(f"{typex} is not supported.")
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

  # return nn.MSELoss()(mu, labels)
  return loss.mean()

# def reduce_ensemble_loss(losses):
#   aleatoric = 
#   epistemic = 


def eval(model, model_type, scramble_batches, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for inputs, labels in tqdm(loader):
        batch_loss = 0.0
        inputs, labels = torch.unsqueeze(inputs, 1).to(device), torch.unsqueeze(labels, 1).to(device)
        outputs = model(inputs)
        if model_type == "shared":
            for head_i, output in enumerate(outputs):
                if scramble_batches:
                  batch_loss += criterion(output, labels[:,:,head_i]) * output.shape[0]
                else:
                  batch_loss += criterion(output, labels) * output.shape[0]
        else:
            batch_loss = criterion(outputs, labels)
        total_loss += batch_loss.item()
    return total_loss / len(loader)

def train(args, device, save_path=None):
    polyf, varf, gaps = make_polyf(args.polyf_type)
    train_dataset = PolyData(polyf, varf, gaps, size=args.train_size, seed=1111, scramble=args.scramble_batches, num_heads=args.num_heads)
    val_dataset = PolyData(polyf, varf, gaps, size=args.val_size, seed=2222, scramble=args.scramble_batches, num_heads=args.num_heads)
    test_dataset = PolyData(polyf, varf, gaps, size=args.test_size, seed=3333, scramble=args.scramble_batches, num_heads=args.num_heads)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    if args.model_type == "single":
        model = ToyNet()
    elif args.model_type == "shared":
        # FIX LAYER SHAPES
        model = VariableBackbone(TOY_LAYER_SHAPES, args.split_idx, args.num_heads, args.scramble_batches)
    model.to(device)

    criterion = nll_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for i in range(args.n_epochs):
        model.train()
        total_train_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            batch_loss = 0.0
            inputs, labels = torch.unsqueeze(inputs, 1).to(device), torch.unsqueeze(labels, 1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if args.model_type == "shared":
                for head_i, output in enumerate(outputs):
                    if args.scramble_batches:
                      batch_loss += criterion(output, labels[:,:,head_i])
                    else:
                      batch_loss += criterion(output, labels)
            else:
                batch_loss = criterion(outputs, labels)
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



