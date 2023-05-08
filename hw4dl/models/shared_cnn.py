import torch
import torch.nn as nn
import pdb
import numpy as np
class VariableCNNBackbone(nn.Module):
    def __init__(self, layer_shapes: tuple, split_idx: int, num_heads: int, kernel_size: int=3, pool_size=2, input_size=(50,50),
                 task="patch"):
        """
        Create a NN with variable amount of shared backbone

        @param layer_shapes: tuple of layer output shapes. The input is assumed to have one channel.
         -1 denotes a Maxpool layer and 'fcx' denotes a FC layer with output size x.
         eg. (32, -1, 16, fc12) makes a network with a convolutional layer with 32 output channels, a max pool layer, a
         conv layer with 16 out channels, and a linear layer with 12 outputs.
        @param split_idx: index of layer to split the backbone. Eg in the above example 2 would split after the
         max pool layer.
        @param num_heads: number of heads to create
        @param kernel_size: kernel dimension for convolutional layers
        @param pool_size: kernel dimension for MaxPool layers
        @input_size

        """
        super().__init__()
        self.layer_shapes = layer_shapes
        self.split_idx = split_idx
        self.num_heads = num_heads
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.task = task
        assert self.split_idx <= len(self.layer_shapes), "Split index must be less than the number of layers"
        shared_backbone = []

        c = 1
        on_fc = False
        input_divide = 1

        splitting = True
        layer_counter, i = 0, 0
        while splitting:
            if layer_counter == self.split_idx or i == len(self.layer_shapes)-1:
                #if type(layer_shapes[i-1]) == str:
                #    c = int(layer_shapes[i-1].split("fc")[1])
                #else:
                #    c = layer_shapes[i-1]
                break
            if layer_shapes[i] == -1:
                shared_backbone.append(nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size))
                input_divide *= self.pool_size
                i += 1
            elif 'fc' in str(layer_shapes[i]):
                if not on_fc:
                    shared_backbone.append(nn.Flatten())
                    on_fc = True
                    c = c * (input_size[0]//input_divide) * (input_size[1]//input_divide)
                out_size = int(layer_shapes[i].split("fc")[1])
                shared_backbone.append(nn.Linear(c, out_size))
                if i != len(self.layer_shapes) - 1:
                    shared_backbone.append(nn.ReLU())
                c = out_size
                layer_counter += 1
                i += 1
            else:
                shared_backbone.append(
                    nn.Conv2d(c, layer_shapes[i], kernel_size=self.kernel_size, stride=1, padding=1))
                if i != len(self.layer_shapes) - 1:
                    shared_backbone.append(nn.ReLU())
                c = layer_shapes[i]
                layer_counter += 1
                i += 1

        self.shared_backbone = nn.Sequential(*shared_backbone)
        head_start_dim = c
        heads = []
        for k in range(self.num_heads):
            head_i = []
            head_input_divide = input_divide
            # Reset c at start of every head
            c = head_start_dim
            head_on_fc = on_fc
            for j in range(i, len(self.layer_shapes)):
                if layer_shapes[j] == -1:
                    head_i.append(nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size))
                    head_input_divide *= self.pool_size
                elif 'fc' in str(layer_shapes[j]):
                    if not head_on_fc:
                        head_i.append(nn.Flatten())
                        head_on_fc = True
                        c = c * (input_size[0] // head_input_divide) * (input_size[1] // head_input_divide)
                    out_size = int(layer_shapes[j].split("fc")[1])
                    head_i.append(nn.Linear(c, out_size))
                    if j != len(self.layer_shapes) - 1:
                        head_i.append(nn.ReLU())
                    c = out_size
                else:
                    head_i.append(
                        nn.Conv2d(c, layer_shapes[j], kernel_size=self.kernel_size, stride=1, padding=1))
                    if j != len(self.layer_shapes) - 1:
                        head_i.append(nn.ReLU())
                    c = layer_shapes[j]

            heads.append(nn.Sequential(*head_i))
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        x = self.shared_backbone(x)
        outputs = []
        for head in self.heads:
            if self.task == "patch":
                outputs.append(head(x).view(-1, 2, 3, 3))
            else:
                outputs.append(head(x))

        return outputs


if __name__ == "__main__":
    layer_shapes = (16, -1, 32, -1, 64, 128, 'fc512', 'fc18')
    num_heads = 5
    for split_idx in range(6):
        print(split_idx)
        x = torch.randn(2, 1, 15, 15)
        model = VariableCNNBackbone(layer_shapes, split_idx, num_heads, input_size=(15,15), task="pixel")
        output = model(x)
        shared_parameters = np.sum([np.prod(p.shape) for p in model.shared_backbone.parameters()])
        print("Shared ", shared_parameters)
        split_parameters = np.sum([np.prod(p.shape) for p in model.heads.parameters()])
        print("Split ", split_parameters)
        print("Total ", shared_parameters + split_parameters)
