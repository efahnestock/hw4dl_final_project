import torch
import torch.nn as nn
import pdb

class VariableCNNBackbone(nn.Module):
    def __init__(self, layer_shapes: tuple, split_idx: int, num_heads: int):
        """
        Create a NN with variable amount of shared backbone

        @param layer_shapes: tuple of layer shapes eg. (input, hidden1, hidden2, output)
        @param split_idx: index of layer to split the backbone. Eg 2 would split after hidden2
        @param num_heads: number of heads to create

        """
        super().__init__()
        self.layer_shapes = layer_shapes
        self.split_idx = split_idx
        self.num_heads = num_heads
        assert self.split_idx < len(self.layer_shapes) - 1, "Split index must be less than the number of layers"
        shared_backbone = []

        c = layer_shapes[0]

        for i in range(1, self.split_idx):
            if layer_shapes[i] == -1:
                shared_backbone.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                shared_backbone.append(
                    nn.Conv2d(c, layer_shapes[i], kernel_size=3, stride=1, padding=1))
                if i != len(self.layer_shapes) - 2:
                    shared_backbone.append(nn.ReLU())
                c = layer_shapes[i]

        self.shared_backbone = nn.Sequential(*shared_backbone)

        heads = []
        for i in range(self.num_heads):
            head_i = []
            for j in range(self.split_idx, len(self.layer_shapes) - 1):
                if layer_shapes[j] == -1:
                    head_i.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    head_i.append(
                        nn.Conv2d(c, layer_shapes[j], kernel_size=3, stride=1, padding=1))
                    if j != len(self.layer_shapes) - 2:
                        head_i.append(nn.ReLU())
                    c = layer_shapes[j]

            heads.append(nn.Sequential(*head_i))
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        x = self.shared_backbone(x)
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        return outputs


if __name__ == "__main__":
    layer_shapes = (16, 16, -1, 32, 32,  64, -1)
    split_idx = 2
    num_heads = 3
    model = VariableCNNBackbone(layer_shapes, split_idx, num_heads)
    print(model)