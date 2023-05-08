import argparse
from hw4dl.train import train
import torch

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_type', type=str, default="cuda", help="type of device to run on; one of [mps, gpu]")
    parser.add_argument('--polyf_type', type=str, default="cubic", help='polynomial fn for instantiating dataset; supports [cubic]')
    parser.add_argument('--train_size', type=int, default=1000, help='Train dataset size')
    parser.add_argument('--val_size', type=int, default=2000, help='Val dataset size')
    parser.add_argument('--test_size', type=int, default=2000, help='Test dataset size')
    parser.add_argument('--batch_size', type=int, default=512, help='Dataset batch size')
    parser.add_argument('--model_type', type=str, default="shared", help="Type of model; one of [shared, single]")
    parser.add_argument('--split_idx', type=int, default=0, help="index of layer to split the backbone for a separated network. Eg 2 would split after hidden2")
    parser.add_argument('--num_heads', type=int, default=5, help="number of prediction heads for a separated network")
    parser.add_argument('--scramble_batches', type=bool, default=False, help="scramble batches for a separated network")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--n_epochs', type=int, default=15, help="number of epochs")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_options()
    if args.device_type == "mps":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    train(args, device)


