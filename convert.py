import torchvision.models as models
import pytorch2timeloop

from models.shared_backbone import VariableBackbone
from models.separated_network import ToyNet

if __name__ == "__main__":
    print(models.alexnet())

    layer_shapes = [1, 20, 30, 40]
    split_idx = 2
    num_heads = 3
    net = VariableBackbone(layer_shapes, split_idx, num_heads)

    input_shape = (1, 1, 1)
    batch_size = 1

    top_dir = 'workspace/final-project/layer_shapes'
    sub_dir = 'ToyNet'

    convert_fc = True

    exception_module_names = []

    pytorch2timeloop.convert_model(net, input_shape, batch_size, sub_dir, top_dir, convert_fc, exception_module_names)
