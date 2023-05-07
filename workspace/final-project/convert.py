import pytorch2timeloop
import os
from hw4dl.models.shared_backbone import VariableBackbone
from hw4dl.models.separated_network import ToyNet
from hw4dl.models.shared_cnn import VariableCNNBackbone
from hw4dl import ROOT_DIR
import yaml
from yaml.loader import SafeLoader
import re
import argparse
import torch


def convert_VariableBackbone(net_params, device, top_dir, model_name, mode=None, params=None):
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
    input_shape = (1, 1, 1)
    batch_size = 1
    convert_fc = True
    exception_module_names = []
    pytorch2timeloop.convert_model(net, input_shape, batch_size, sub_dir, top_dir, convert_fc, exception_module_names, params)

    if mode == 'parallel':
        workloads_per_head = (len(layer_shapes) - 1) - split_idx
        param_dir = get_param_name(model_name, params)
        # get layer files and sort by layer number
        layer_dir = os.path.join(top_dir, sub_dir)
        if param_dir:
            layer_dir = os.path.join(layer_dir, param_dir)
        layer_files = os.listdir(layer_dir)
        layer_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

        # delete all but one head
        layer_files_del = layer_files[-(num_heads-1)*workloads_per_head:]
        for l in layer_files_del:
            os.remove(os.path.join(layer_dir, l))
        layer_files = layer_files[:-(num_heads-1)*workloads_per_head]

        # add index for heads to Einsum where appropriate
        for l in range(len(layer_files)-workloads_per_head,len(layer_files)):
            print(len(layer_files)-workloads_per_head,len(layer_files))
            with open(os.path.join(layer_dir, layer_files[l]), 'r') as f:
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
            with open(os.path.join(layer_dir, layer_files[l]), 'w') as f:
                yaml.safe_dump(problem, f)
    else:
        return


def convert_VariableCNNBackbone(net_params, device, top_dir, model_name, mode=None, params=None):
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
    print(layer_shapes)
    net = VariableCNNBackbone(layer_shapes, split_idx, num_heads, input_size=(10,10), task='pixel').to(device)
    print(net)

    # pytorch2timeloop
    sub_dir = 'VariableCNNBackbone'
    input_shape = (1, 10, 10)
    batch_size = 1
    convert_fc = True
    exception_module_names = []

    pytorch2timeloop.convert_model(net, input_shape, batch_size, sub_dir, top_dir, convert_fc, exception_module_names, params)

    if mode == 'parallel':
        workloads_per_head = len([i for i in layer_shapes if i != -1]) - split_idx
        param_dir = get_param_name(model_name, params)
        # get layer files and sort by layer number
        layer_dir = os.path.join(top_dir, sub_dir)
        if param_dir:
            layer_dir = os.path.join(layer_dir, param_dir)
        layer_files = os.listdir(layer_dir)
        layer_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

        # delete all but one head
        layer_files_del = layer_files[-(num_heads-1)*workloads_per_head:]
        for l in layer_files_del:
            os.remove(os.path.join(layer_dir, l))
        layer_files = layer_files[:-(num_heads-1)*workloads_per_head]

        # add index for heads to Einsum where appropriate
        for l in range(len(layer_files)-workloads_per_head,len(layer_files)):
            print(len(layer_files)-workloads_per_head,len(layer_files))
            with open(os.path.join(layer_dir, layer_files[l]), 'r') as f:
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
            with open(os.path.join(layer_dir, layer_files[l]), 'w') as f:
                yaml.safe_dump(problem, f)
    else:
        return


def convert_ToyNet(net_params, device, top_dir, params=None):
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
    pytorch2timeloop.convert_model(net, input_shape, batch_size, sub_dir, top_dir, convert_fc, exception_module_names, params)


def get_param_name(model_name, params):
    print(model_name)
    if 'ToyNet' in model_name:
        name = '%slayers_%sshape' % (str(params['num_layers']), "-".join(str(x) for x in params['layer_shapes']))
    elif 'VariableBackbone' or 'VariableCNNBackbone' in model_name:
        name = '%sshape_%ssplit_%sheads_%s' % ("-".join(str(x) for x in params['layer_shapes']), str(params['split_idx']), str(params['num_heads']), params['mode'])
    else:
        name = None
    return name


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help='Name of params yaml')
    parser.add_argument('--mode', type=str, default="serial", help="Process heads: serial, parallel")
    parser.add_argument('--model_type', type=str, default="VariableBackbone", help="Name of model")
    parser.add_argument('--top_dir', type=str, default="layer_shapes", help="Directory with layer shapes")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_options()
    # results top directory
    args.top_dir = os.path.join(ROOT_DIR, "workspace/final-project/layer_shapes")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    with open(f"configs/{args.params}.yaml", 'r') as f:
        params = yaml.safe_load(f)

    if args.model_type == 'ToyNet':
        convert_ToyNet(params, device, args.top_dir, params=params)
    elif args.model_type == 'VariableBackbone':
        convert_VariableBackbone(params, device, args.top_dir, args.model_type, mode=params['mode'], params=params)
    elif args.model_type == 'VariableCNNBackbone':
        convert_VariableCNNBackbone(params, device, args.top_dir, args.model_type, mode=params['mode'], params=params)
