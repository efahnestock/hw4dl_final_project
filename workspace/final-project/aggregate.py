import argparse
import yaml
import os


def aggregate(args):
    with open(f"configs/{args.params}.yaml", 'r') as f:
        params = yaml.safe_load(f)
    timeloop_dir = os.path.join('timeloop_results', args.design)
    sub_dir = args.model_type

    param_dir = get_param_name(args.model_type, params)
    layer_dir = os.path.join(args.base_dir, args.top_dir, sub_dir)
    if param_dir:
        layer_dir = os.path.join(layer_dir, param_dir)
    layer_files = os.listdir(layer_dir)

    # Gather layers recorded in Timeloop
    layer_info = {}
    for file in os.listdir(os.path.join(args.base_dir, timeloop_dir, sub_dir, param_dir)):
        with open(os.path.join(layer_dir, file+'.yaml'), 'r') as fid:
            layer_dict = yaml.safe_load(fid)
            idx = int(file.replace('layer', ''))
            layer_info[idx] = {
                'layer_dict': layer_dict,
                'num': 1
            }

    # Get duplicated layer info
    for idx, file in enumerate(layer_files):
        with open(os.path.join(layer_dir, file), 'r') as fid:
            layer_dict = yaml.safe_load(fid)
            for layer_id, info in layer_info.items():
                if info['layer_dict'] == layer_dict:
                    layer_info[layer_id]['num'] += 1

    total_energy, total_cycles = 0, 0
    for layer_id in layer_info.keys():
        with open(os.path.join(args.base_dir, timeloop_dir, sub_dir, param_dir, f'layer{layer_id}', f'timeloop-mapper.stats.txt'), 'r') as fid:
            # Read last 50 lines
            lines = fid.read().split('\n')[-50:]
            for line in lines:
                if line.startswith('Energy'):
                    energy = line.split(': ')[1].split(' ')[0]
                    total_energy += eval(energy)*layer_info[layer_id]['num']
                    layer_info[layer_id]['energy'] = eval(energy)
                elif line.startswith('Cycles'):
                    cycle = line.split(': ')[1]
                    total_cycles += eval(cycle)*layer_info[layer_id]['num']
                    layer_info[layer_id]['cycle'] = eval(cycle)
                elif line.startswith('GFLOPs'):
                    gflops = line.split(': ')[1]
                    layer_info[layer_id]['gflops'] = eval(gflops)
                elif line.startswith('Utilization'):
                    ut = line.split(': ')[1]
                    layer_info[layer_id]['utilization'] = eval(ut)

    print('%f uJ Energy' % total_energy)
    print('%d Cycles' % total_cycles)
    return


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
    parser.add_argument('--params', type=str, help='Name of params yaml')
    parser.add_argument('--base_dir', type=str, help='Base directory')
    parser.add_argument('--top_dir', type=str, default="layer_shapes", help="Directory with layer shapes")
    parser.add_argument('--design', type=str, default="eyeriss_like", help="Architecture design")
    parser.add_argument('--model_type', type=str, default="VariableBackbone", help="Name of model")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_options()
    aggregate(args)
