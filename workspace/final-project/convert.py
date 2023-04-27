import pytorch2timeloop
import os
import torch
from hw4dl.models.shared_backbone import VariableBackbone
from hw4dl.models.separated_network import ToyNet
from hw4dl.models.shared_cnn import VariableCNNBackbone
from hw4dl import ROOT_DIR
import yaml
from yaml.loader import SafeLoader
import re

def convert_VariableBackbone(net_params, device, top_dir, mode=None):
    """
    Function to convert the VariableBackbone model to timeloop problem descriptions

    Parameters
    ----------
    net_params : dict
        dictionary containing the keys 'layer_shapes', 'split_idx', and 'num_heads' for
        the VariableBackbone model
    device : str
        device to put the model on
    top_dir : str
        path to directory with all layer shape directories
    mode : str
        choose 'serial' or 'parallel'
    """

    assert mode in ['serial', 'parallel'], "mode must be one of 'serial' or 'parallel'"

    # extract network parameters
    layer_shapes = net_params['layer_shapes']
    split_idx = net_params['split_idx']
    num_heads = net_params['num_heads']

    if mode and num_heads == 1:
        raise ValueError("If only one head, don't use the mode argument.")

    # make network
    net = VariableBackbone(layer_shapes, split_idx, num_heads).to(device)

    # pytorch2timeloop
    sub_dir = 'VariableBackbone'
    if mode:
        sub_dir = sub_dir + f'_{mode}'
    input_shape = (1, 1, 1)
    batch_size = 1
    convert_fc = True
    exception_module_names = []
    pytorch2timeloop.convert_model(net, input_shape, batch_size, sub_dir, top_dir, convert_fc, exception_module_names)

    if mode == 'parallel':

        # get layer files and sort by layer number
        layer_dir = os.path.join(top_dir, sub_dir)
        layer_files = os.listdir(layer_dir)
        layer_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

        # delete all but one head
        layer_files_del = layer_files[-num_heads+1:]
        for l in layer_files_del:
            os.remove(os.path.join(layer_dir, l))
        layer_files = layer_files[:-num_heads+1]
        
        # add index for heads to Einsum where appropriate
        with open(os.path.join(layer_dir, layer_files[-1]), 'r') as f:
            problem = yaml.load(f, Loader=SafeLoader)
            
            # update in instance
            problem['problem']['instance']['H'] = num_heads
            
            # update in dimensions
            problem['problem']['shape']['dimensions'].append('H')

            # update in data-spaces
            data_spaces = problem['problem']['shape']['data-spaces']
            for i, ds in enumerate(data_spaces):
                if ds['name'] == 'Weights' or ds['name'] == 'Outputs':
                    problem['problem']['shape']['data-spaces'][i]['projection'].append([['H']])

        # write changes
        with open(os.path.join(layer_dir, layer_files[-1]), 'w') as f:
            yaml.safe_dump(problem, f)
    else:
        return

def convert_VariableCNNBackbone(net_params, device, top_dir, mode=None):
    """
    Function to convert the VariableBackbone model to timeloop problem descriptions

    Parameters
    ----------
    net_params : dict
        dictionary containing the keys 'layer_shapes', 'split_idx', and 'num_heads' for
        the VariableBackbone model
    device : str
        device to put the model on
    top_dir : str
        path to directory with all layer shape directories
    mode : str
        choose 'serial' or 'parallel'
    """

    assert mode in ['serial', 'parallel'], "mode must be one of 'serial' or 'parallel'"

    # extract network parameters
    layer_shapes = net_params['layer_shapes']
    split_idx = net_params['split_idx']
    num_heads = net_params['num_heads']

    if mode and num_heads == 1:
        raise ValueError("If only one head, don't use the mode argument.")

    # make network
    net = VariableCNNBackbone(layer_shapes, split_idx, num_heads).to(device)

    # pytorch2timeloop
    sub_dir = 'VariableBackbone'
    if mode:
        sub_dir = sub_dir + f'_{mode}'
    input_shape = (3, 224, 224)
    batch_size = 1
    convert_fc = True
    exception_module_names = []
    pytorch2timeloop.convert_model(net, input_shape, batch_size, sub_dir, top_dir, convert_fc, exception_module_names)

    if mode == 'parallel':

        # get layer files and sort by layer number
        layer_dir = os.path.join(top_dir, sub_dir)
        layer_files = os.listdir(layer_dir)
        layer_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

        # delete all but one head
        layer_files_del = layer_files[-num_heads+1:]
        for l in layer_files_del:
            os.remove(os.path.join(layer_dir, l))
        layer_files = layer_files[:-num_heads+1]
        
        # add index for heads to Einsum where appropriate
        with open(os.path.join(layer_dir, layer_files[-1]), 'r') as f:
            problem = yaml.load(f, Loader=SafeLoader)
            
            # update in instance
            problem['problem']['instance']['H'] = num_heads
            
            # update in dimensions
            problem['problem']['shape']['dimensions'].append('H')

            # update in data-spaces
            data_spaces = problem['problem']['shape']['data-spaces']
            for i, ds in enumerate(data_spaces):
                if ds['name'] == 'Weights' or ds['name'] == 'Outputs':
                    problem['problem']['shape']['data-spaces'][i]['projection'].append([['H']])

        # write changes
        with open(os.path.join(layer_dir, layer_files[-1]), 'w') as f:
            yaml.safe_dump(problem, f)
    else:
        return

def convert_ToyNet(net_params, device, top_dir):
    """
    Function to convert the ToyNet model to timeloop problem descriptions

    Parameters
    ----------
    net_params : dict
        dictionary containing the keys 'num_layers', and 'layer_shapes' for
        the VariableBackbone model. The first layer shape must be 1.
    device : str
        device to put the model on
    top_dir : str
        path to directory with all layer shape directories
    """
    
    # extract parameters
    num_layers = net_params['num_layers']
    layer_shapes = net_params['layer_shapes']

    # make network
    net = ToyNet(num_layers, layer_shapes).to(device)

    # pytorch2timeloop
    sub_dir = 'ToyNet'
    input_shape = (1, 1, 1)
    batch_size = 1
    convert_fc = True
    exception_module_names = []
    pytorch2timeloop.convert_model(net, input_shape, batch_size, sub_dir, top_dir, convert_fc, exception_module_names)

if __name__ == "__main__":

    # results top directory
    top_dir = os.path.join(ROOT_DIR, "workspace/final-project/layer_shapes")

    # VariableBackbone
    convert_VariableBackbone(net_params={'layer_shapes' : [1, 20, 30, 40],
                                         'split_idx' : 2,
                                         'num_heads' : 3},
                             device='cuda:0',
                             top_dir=top_dir,
                             mode='parallel')

    convert_VariableCNNBackbone(net_params={'layer_shapes' : (3, 16, -1, 32, 32, 64, -1),
                                            'split_idx' : 2,
                                            'num_heads' : 3},
                                device='cuda:0',
                                top_dir=top_dir,
                                mode='parallel')

    # ToyNet
    convert_ToyNet(net_params={'layer_shapes' : [1, 20, 30, 40],
                               'num_layers' : 3},
                               device='cuda:0',
                               top_dir=top_dir)