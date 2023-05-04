import os
import yaml
from tqdm import tqdm
from pathlib import Path
import argparse
from pytimeloop.app import ModelApp, MapperApp
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO


class Profiler(object):
    def __init__(self,
                 base_dir,
                 sub_dir,
                 top_dir,
                 timeloop_dir,
                 model,
                 params,
                 design,
                 input_size,
                 batch_size,
                 convert_fc,
                 exception_module_names
                 ):
        self.base_dir = base_dir
        self.sub_dir = sub_dir
        self.top_dir = top_dir
        self.model = model
        self.params = params
        self.design = design
        self.timeloop_dir = timeloop_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.convert_fc = convert_fc
        self.exception_module_names = exception_module_names

    def profile(self) -> dict:
        self.param_dir = get_param_name(self.model, self.params)
        layer_dir = self.base_dir / self.top_dir / self.sub_dir
        if self.param_dir:
            layer_dir = layer_dir / self.param_dir
        layer_files = os.listdir(layer_dir)

        # Check duplicated layer info
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

        # Run timeloop mapper
        for layer_id in layer_info.keys():
            os.makedirs(self.base_dir / self.timeloop_dir / self.sub_dir / self.param_dir / f'layer{layer_id}', exist_ok=True)

        def get_cmd(layer_id):
            cwd = f"{self.base_dir / self.timeloop_dir / self.sub_dir / self.param_dir / f'layer{layer_id}'}"

            timeloopcmd = f"timeloop-mapper " \
                          f"{self.base_dir / self.timeloop_dir / 'arch' / f'{args.design}.yaml'} " \
                          f"{self.base_dir / self.timeloop_dir / 'arch/components/*.yaml'} " \
                          f"{self.base_dir / self.timeloop_dir / 'mapper/mapper.yaml'} " \
                          f"{self.base_dir / self.timeloop_dir / 'constraints/*.yaml'} " \
                          f"{self.base_dir / self.top_dir / self.sub_dir / self.param_dir / f'layer{layer_id}.yaml'}"
            print(timeloopcmd)
            return [cwd, timeloopcmd]

        cmds_list = list(map(get_cmd, layer_info.keys()))

        for cwd, cmd in tqdm(cmds_list):
            os.chdir(cwd)
            os.system(cmd)
        os.chdir(self.base_dir)

        print(f'timeloop running finished!')

        """
        for idx, file in enumerate(layer_files):
            stats, loops = run_timeloop_mapper(
                Path(f"{self.base_dir/self.timeloop_dir/'arch'/f'{args.design}.yaml'}"),
                Path(f"{self.base_dir/self.timeloop_dir/'arch/components/'}"),
                Path(f"{self.base_dir/self.timeloop_dir/'constraints/*.yaml'}"),
                Path(f"{self.base_dir/self.timeloop_dir/'mapper/mapper.yaml'}"),
                Path(f"{self.base_dir/args.top_dir/self.sub_dir/f'{file}'}")
            )
            with open(self.base_dir/self.timeloop_dir/self.sub_dir/f'layer{idx+1}'/f'timeloop-mapper.stats.txt', 'w') as fid:
                fid.write(stats)
        """

        # Collect stats
        for layer_id in layer_info.keys():
            with open(self.base_dir / self.timeloop_dir / self.sub_dir / self.param_dir / f'layer{layer_id}' / f'timeloop-mapper.stats.txt', 'r') as fid:
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


def get_param_name(model_name, params):
    print(model_name)
    if 'ToyNet' in model_name:
        name = '%slayers_%sshape' % (str(params['num_layers']), "-".join(str(x) for x in params['layer_shapes']))
    elif 'VariableBackbone' in model_name:
        name = '%sshape_%ssplit_%sheads_%s' % ("-".join(str(x) for x in params['layer_shapes']), str(params['split_idx']), str(params['num_heads']), params['mode'])
    else:
        name = None
    return name


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=tuple, default=(1,1,1), help='Data example size')
    parser.add_argument('--batch_size', type=int, default=256, help='Dataset batch size')
    parser.add_argument('--model_type', type=str, default="VariableBackbone", help="Name of model")
    parser.add_argument('--base_dir', type=str, help='Base directory')
    parser.add_argument('--top_dir', type=str, default="layer_shapes", help="Directory with layer shapes")
    parser.add_argument('--design', type=str, default="simple_weight_stationary", help="Architecture design")
    parser.add_argument('--params', type=str, default=None, help='Name of params yaml')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_options()
    if args.params:
        with open(f"configs/{args.params}.yaml", 'r') as f:
            params = yaml.safe_load(f)
    else:
        params = None

    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path(os.getcwd())

    profiler = Profiler(
        base_dir=base_dir,
        top_dir=args.top_dir,
        sub_dir=args.model_type,
        timeloop_dir=os.path.join('timeloop_results', args.design),
        model=args.model_type,
        params=params,
        design=args.design,
        input_size=args.input_size,
        batch_size=args.batch_size,
        exception_module_names=[],
        convert_fc=True
    )
    results = profiler.profile()
    print(results)
