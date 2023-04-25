import pytorch2timeloop
import os
import torch
from hw4dl.models.shared_backbone import VariableBackbone
from hw4dl.models.separated_network import ToyNet
import yaml

def convert_VariableBackbone(net_params, device, top_dir):

    # extract network parameters
    layer_shapes = net_params['layer_shapes']
    split_idx = net_params['split_idx']
    num_heads = net_params['num_heads']

    # make network
    net = VariableBackbone(layer_shapes, split_idx, num_heads).to(device)

    sub_dir = 'ToyNet'
    input_shape = (1, 1, 1)
    batch_size = 1
    convert_fc = True
    exception_module_names = []
    pytorch2timeloop.convert_model(net, input_shape, batch_size, sub_dir, top_dir, convert_fc, exception_module_names)


if __name__ == "__main__":

    # results top directory
    top_dir = 'layer_shapes'

    # VariableBackbone
    convert_VariableBackbone(net_params={'layer_shapes' : [1, 20, 30, 40],
                                         'split_idx' : 2,
                                         'num_heads' : 3},
                             device='cuda:0',
                             top_dir=top_dir)