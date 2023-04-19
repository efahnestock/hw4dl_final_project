from loaders.toy_loader import PolyData
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from models.separated_network import ToyNet
from models.shared_backbone import VariableBackbone
import numpy as np
import pdb
TOY_LAYER_SHAPES = [1, 10, 20, 10, 1]

def make_polyf(typex):
    """
    Return a polynomial function, and associated variance function and gaps for constructing a dataset
    """
    if typex == "cubic":
        def polyf(x):
            return 2*x ** 3 + 5*x ** 2 + 2

        def varf(x):
            return 0.5*x + 0.5

        gaps = [(0.5, 0.6), (0.1, 0.3)]
        return polyf, varf, gaps
    else:
        raise ValueError(f"{typex} is not supported.")

def eval(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for inputs, labels in tqdm(loader):
        batch_loss = 0.0
        inputs, labels = torch.unsqueeze(inputs, 1).to(device), torch.unsqueeze(labels, 1).to(device)
        outputs = model(inputs)
        for output in outputs:
            batch_loss += criterion(output, labels)
        total_loss += batch_loss.item()
    return total_loss / len(loader)

def train(args, device):
    polyf, varf, gaps = make_polyf(args.polyf_type)
    train_dataset = PolyData(polyf, varf, gaps, size=args.train_size, seed=1111)
    val_dataset = PolyData(polyf, varf, gaps, size=args.val_size, seed=2222)
    test_dataset = PolyData(polyf, varf, gaps, size=args.test_size, seed=3333)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    if args.model_type == "separate":
        model = ToyNet()
    elif args.model_type == "shared":
        # FIX LAYER SHAPES
        model = VariableBackbone(TOY_LAYER_SHAPES, args.split_idx, args.num_heads)
    model.to(device)

    criterion = nn.MSELoss()
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
                for output in outputs:
                    batch_loss += criterion(output, labels)
            else:
                batch_loss = criterion(output, labels)
            batch_loss.backward()
            optimizer.step()
            total_train_loss += batch_loss.item()

        train_loss = total_train_loss / len(train_loader)
        val_loss = eval(model, val_loader, criterion, device)
        print(f"Epoch {i}, train loss: {train_loss}, val loss: {val_loss}")

    test_loss = eval(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss}")

    print("Done :)")



