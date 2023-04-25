import torch.nn as nn


class ToyNet(nn.Module):
    def __init__(self, num_layers, layer_shapes):
        super(ToyNet, self).__init__()
        network = []
        for i in range(num_layers):
            network.append(nn.Linear(layer_shapes[i], layer_shapes[i+1]))
            if i != num_layers-1:
                network.append(nn.ReLU(inplace=True))
        self.network = nn.Sequential(*network)

    def forward(self, x):
        z = self.network(x)
        return z


if __name__ == "__main__":
    layer_shapes = [10, 20, 30, 40]
    num_layers = 3
    model = ToyNet(num_layers, layer_shapes)
    print(model)
