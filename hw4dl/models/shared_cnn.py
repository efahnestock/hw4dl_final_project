import torch
import torch.nn as nn
import pdb

class VariableCNNBackbone(nn.Module):
    def __init__(self, layer_shapes: tuple, split_idx: int, num_heads: int, kernel_size: int=3, pool_size=2, input_size=(50,50),
                 task="patch"):
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
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.task = task
        assert self.split_idx <= len(self.layer_shapes), "Split index must be less than the number of layers"
        shared_backbone = []

        c = layer_shapes[0]
        on_fc = False
        input_divide = 1

        for i in range(1, self.split_idx):
            if layer_shapes[i] == -1:
                shared_backbone.append(nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size))
                input_divide *= self.pool_size
            elif 'fc' in str(layer_shapes[i]):
                if not on_fc:
                    shared_backbone.append(nn.Flatten())
                    on_fc = True
                    c = c * input_size[0]//input_divide * input_size[1]//input_divide
                out_size = int(layer_shapes[i].split("fc")[1])
                shared_backbone.append(nn.Linear(c, out_size))
                if i != len(self.layer_shapes) - 1:
                    shared_backbone.append(nn.ReLU())
                c = out_size
            else:
                shared_backbone.append(
                    nn.Conv2d(c, layer_shapes[i], kernel_size=self.kernel_size, stride=1, padding=1))
                if i != len(self.layer_shapes) - 1:
                    shared_backbone.append(nn.ReLU())
                c = layer_shapes[i]

        self.shared_backbone = nn.Sequential(*shared_backbone)
        head_start_dim = c
        heads = []
        for i in range(self.num_heads):
            head_i = []
            head_input_divide = input_divide
            c = head_start_dim
            head_on_fc = on_fc
            for j in range(self.split_idx, len(self.layer_shapes)):
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
    layer_shapes = (1, 16, -1, 32, -1, 64, 'fc512', 'fc18')
    split_idx = 2
    num_heads = 3
    x = torch.randn(2, 1, 10, 10)
    model = VariableCNNBackbone(layer_shapes, split_idx, num_heads, input_size=(10,10), task="pixel")
    print(model)
    output = model(x)
    print(len(output))
    print(output[0].shape, output[1].shape, output[2].shape)