import os
import yaml
from tqdm import tqdm
from pathlib import Path
import argparse
from convert import convert_VariableBackbone


class Profiler(object):
    def __init__(self,
                 sub_dir,
                 top_dir,
                 timeloop_dir,
                 model,
                 input_size,
                 batch_size,
                 convert_fc,
                 exception_module_names
                 ):
        self.base_dir = Path(os.getcwd())
        self.sub_dir = sub_dir
        self.top_dir = top_dir
        self.model = model
        self.timeloop_dir = timeloop_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.convert_fc = convert_fc
        self.exception_module_names = exception_module_names

    def profile(self) -> dict:
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
            f"{self.base_dir/self.timeloop_dir/'arch/simple_weight_stationary.yaml'} " \
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



def get_model(model_type):
    if model_type == 'VariableBackbone':
        model = convert_VariableBackbone()
    return model


def benchmark(args):
    model = get_model(args.model_type)
    profiler = Profiler(
        top_dir='workloads',
        sub_dir=args.model_type,
        timeloop_dir='simple_weight_stationary',
        model=model,
        input_size=args.input_size,
        batch_size=args.batch_size,
        exception_module_names=[],
        convert_fc=True
    )
    profiler.profile()


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=tuple, default=(1,1,1), help='Data example siz')
    parser.add_argument('--batch_size', type=int, default=256, help='Dataset batch size')
    parser.add_argument('--model_type', type=str, default="VariableBackbone", help="Name of model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_options()
    benchmark(args)
