import os
import yaml
from tqdm import tqdm
from pathlib import Path
import argparse
from convert import convert_VariableBackbone, convert_ToyNet
from pytimeloop.app import ModelApp, MapperApp
import torch
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO


class Profiler(object):
    def __init__(self,
                 sub_dir,
                 top_dir,
                 timeloop_dir,
                 model,
                 design,
                 input_size,
                 batch_size,
                 convert_fc,
                 exception_module_names
                 ):
        self.base_dir = Path(os.getcwd())
        self.sub_dir = sub_dir
        self.top_dir = top_dir
        self.model = model
        self.design = design
        self.timeloop_dir = timeloop_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.convert_fc = convert_fc
        self.exception_module_names = exception_module_names

    def profile(self) -> dict:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        convert_model(args.model_type, device, args.top_dir)

        layer_dir = self.base_dir/self.top_dir/self.sub_dir
        layer_files = os.listdir(layer_dir)

        # check duplicated layer info
        layer_info = {}
        for idx, file in enumerate(layer_files):
            with open(layer_dir/file, 'r') as fid:
                layer_dict = yaml.safe_load(fid)
                for layer_id, info in layer_info.items():
                    if info['layer_dict'] == layer_dict:
                        layer_info[layer_id]['num'] += 1
                        break
                else:
                    layer_info[idx+1] = {
                        'layer_dict': layer_dict,
                        'num': 1
                    }

        # run timeloop
        print(f'running timeloop to get energy and latency...')
        for layer_id in layer_info.keys():
            os.makedirs(self.base_dir/self.timeloop_dir/self.sub_dir/f'layer{layer_id}', exist_ok=True)

        def get_cmd(layer_id):
            cwd = f"{self.base_dir/self.timeloop_dir/self.sub_dir/f'layer{layer_id}'}"

            timeloopcmd = f"timeloop-mapper " \
            f"{self.base_dir/self.timeloop_dir/f'arch/{self.design}.yaml'} " \
            f"{self.base_dir/self.timeloop_dir/'arch/components/*.yaml'} " \
            f"{self.base_dir/self.timeloop_dir/'mapper/mapper.yaml'} " \
            f"{self.base_dir/self.timeloop_dir/'constraints/*.yaml'} " \
            f"{self.base_dir/self.top_dir/self.sub_dir/self.sub_dir}_layer{layer_id}.yaml > /dev/null 2>&1"
            return [cwd, timeloopcmd]

        cmds_list = list(map(get_cmd, layer_info.keys()))

        for cwd, cmd in tqdm(cmds_list):
            os.chdir(cwd)
            os.system(cmd)
        os.chdir(self.base_dir)

        print(f'timeloop running finished!')

        for layer_id in layer_info.keys():
            with open(self.base_dir/self.timeloop_dir/self.sub_dir/f'layer{layer_id}'/f'timeloop-mapper.stats.txt', 'r') as fid:
                lines = fid.read().split('\n')[-50:]
                for line in lines:
                    if line.startswith('Energy'):
                        energy = line.split(': ')[1].split(' ')[0]
                        layer_info[layer_id]['energy'] = eval(energy)
                    elif line.startswith('Area'):
                        area = line.split(': ')[1].split(' ')[0]
                        layer_info[layer_id]['area'] = eval(area)
                    elif line.startswith('Cycles'):
                        cycle = line.split(': ')[1]
                        layer_info[layer_id]['cycle'] = eval(cycle)

        return layer_info


"""
Converter
"""


def convert_model(model_type, device, top_dir):
    """
    Converts model from PyTorch to Timeloop
    :return:
    """
    with open(f"configs/{model_type}.yaml", 'r') as f:
        params = yaml.safe_load(f)
    print(params)
    if model_type == 'ToyNet':
        convert_ToyNet(params, device, top_dir)
    elif model_type == 'VariableBackbone':
        convert_VariableBackbone(params, device, top_dir)
    return


"""
Mapper
"""


def dump_str(yaml_dict):
    yaml = YAML(typ='safe')
    yaml.version = (1, 2)
    yaml.default_flow_style = False
    stream = StringIO()
    yaml.dump(yaml_dict, stream)
    return stream.getvalue()


def load_config(*paths):
    yaml = YAML(typ='safe')
    yaml.version = (1, 2)
    total = None
    def _collect_yaml(yaml_str, total):
        new_stuff = yaml.load(yaml_str)
        if total is None:
            return new_stuff

        for key, value in new_stuff.items():
            if key == 'compound_components' and key in total:
                total['compound_components']['classes'] += value['classes']
            elif key in total:
                raise RuntimeError(f'overlapping key: {key}')
            else:
                total[key] = value
        return total

    for path in paths:
        if isinstance(path, str):
            total = _collect_yaml(path, total)
            continue
        elif path.is_dir():
            for p in path.glob('*.yaml'):
                with p.open() as f:
                    total = _collect_yaml(f.read(), total)
        else:
            with path.open() as f:
                total = _collect_yaml(f.read(), total)
    return total


def run_timeloop_mapper(*paths):
    yaml_str = dump_str(load_config(*paths))
    mapper = MapperApp(yaml_str, '.')
    result = mapper.run_subprocess()
    return result


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=tuple, default=(1,1,1), help='Data example siz')
    parser.add_argument('--batch_size', type=int, default=256, help='Dataset batch size')
    parser.add_argument('--model_type', type=str, default="VariableBackbone", help="Name of model")
    parser.add_argument('--top_dir', type=str, default="layer_shapes", help="Directory with layer shapes")
    parser.add_argument('--design', type=str, default="simple_weight_stationary", help="Architecture design")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_options()
    profiler = Profiler(
        top_dir=args.top_dir,
        sub_dir=args.model_type,
        timeloop_dir='timeloop_results',
        model=args.model_type,
        design=args.design,
        input_size=args.input_size,
        batch_size=args.batch_size,
        exception_module_names=[],
        convert_fc=True
    )
    profiler.profile()
