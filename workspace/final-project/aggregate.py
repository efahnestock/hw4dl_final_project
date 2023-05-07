import argparse
import yaml
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import re


def aggregate(args, params):
    designs = []
    for design in os.listdir(os.path.join(args.base_dir, 'timeloop_results')):
        if args.design in design:
            designs.append(design)
    total_energy, total_cycles, total_energy_per, total_ifmap_spad = {}, {}, {}, {}
    for design in designs:
        pes = re.findall(r'\d+', design)[0]
        timeloop_dir = os.path.join('timeloop_results', design)
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

        total_energy_design, total_cycles_design, total_energy_per_design, total_ifmap_spad_design = 0, 0, 0, 0
        for layer_id in layer_info.keys():
            with open(os.path.join(args.base_dir, timeloop_dir, sub_dir, param_dir, f'layer{layer_id}', f'timeloop-mapper.stats.txt'), 'r') as fid:
                # Read last 50 lines
                lines = fid.read().split('\n')[-30:]
                for line in lines:
                    if line.startswith('Energy'):
                        energy = line.split(': ')[1].split(' ')[0]
                        total_energy_design += eval(energy)*layer_info[layer_id]['num']
                        layer_info[layer_id]['energy'] = eval(energy)
                    elif line.startswith('    Total'):
                        energy_per = line.split('= ')[1].split(' ')[0]
                        total_energy_per_design += eval(energy_per)*layer_info[layer_id]['num']
                        layer_info[layer_id]['energy_per'] = eval(energy_per)
                    elif line.startswith('    ifmap_spad'):
                        ifmap_spad = line.split('= ')[1].split(' ')[0]
                        total_ifmap_spad_design += eval(ifmap_spad)*layer_info[layer_id]['num']
                        layer_info[layer_id]['ifmap_spad'] = ifmap_spad
                    elif line.startswith('Cycles'):
                        cycle = line.split(': ')[1]
                        total_cycles_design += eval(cycle)*layer_info[layer_id]['num']
                        layer_info[layer_id]['cycle'] = eval(cycle)
                    elif line.startswith('GFLOPs'):
                        gflops = line.split(': ')[1]
                        layer_info[layer_id]['gflops'] = eval(gflops)
                    elif line.startswith('Utilization'):
                        ut = line.split(': ')[1]
                        layer_info[layer_id]['utilization'] = eval(ut)

        print('%f uJ Energy' % total_energy_design)
        print('%f pJ Energy Per' % total_energy_per_design)
        print('%d Cycles' % total_cycles_design)
        print("%f IFMAP" % total_ifmap_spad_design)
        total_energy[pes] = total_energy_design
        total_cycles[pes] = total_cycles_design
        total_energy_per[pes] = total_energy_per_design
        total_ifmap_spad[pes] = total_ifmap_spad_design
    return total_energy, total_cycles, total_energy_per, total_ifmap_spad


def plot(args, config_dir, performance_csv):
    perform_results = pd.read_csv(performance_csv)
    # Collect results
    results = {
        'backbone size': [],
        'energy': [],
        'cycles': [],
        'performance': [],
        'process': [],
        'energy per': [],
        'ifmap spad': [],
        'pes': []
    }
    for config in os.listdir(config_dir):
        if 'serial' in config:
            process = 'Serial'
        elif 'parallel' in config:
            process = 'Parallel'
        if args.constraints and args.constraints in config:
            continue
        with open(f"{config_dir}/{config}", 'r') as f:
            config_file = yaml.safe_load(f)
        total_energy, total_cycles, total_energy_per, total_ifmap_spad = aggregate(args, config_file)
        for key in total_energy.keys():
            print(perform_results)
            results['backbone size'].append(config_file['split_idx'])
            results['energy'].append(total_energy[key])
            results['cycles'].append(total_cycles[key])
            results['performance'].append(perform_results.at[config_file['split_idx']-1, 'epi_score'])
            results['process'].append(process)
            results['energy per'].append(total_energy_per[key])
            results['ifmap spad'].append(total_ifmap_spad[key])
            results['pes'].append(key)

    # Plot
    df = pd.DataFrame.from_dict(results)
    df['pes'] = df['pes'].astype('int')
    print(df)
    # Energy vs. backbone size
    energy_backbone = sns.lmplot(
        data=df[df['pes']==168], x='backbone size', y='energy', hue='process'
    )
    plt.xlabel('Backbone size (number of layers)')
    plt.ylabel('Energy (uJ)')
    if 'VariableCNNBackbone' in config_dir:
        plt.title('Variable CNN Backbone Size vs. Energy, 168 PEs')
    elif 'VariableBackbone' in config_dir:
        plt.title('FC Backbone Size vs. Energy, 168 PEs')
    energy_backbone.fig.savefig('energy_backbone.png', bbox_inches="tight")
    # Energy vs. performance
    energy_performance = sns.lmplot(
        data=df[df['pes']==168], x='energy', y='performance', hue='process'
    )
    plt.xlabel('Energy (uJ)')
    plt.ylabel('Epistemic uncertainty')
    if 'VariableCNNBackbone' in config_dir:
        plt.title('Variable CNN Energy vs. Performance, 168 PEs')
    elif 'VariableBackbone' in config_dir:
        plt.title('FC Energy vs. Performance, 168 PEs')
    energy_performance.fig.savefig('energy_performance.png', bbox_inches="tight")
    # Cycles vs. backbone size
    cycles_backbone = sns.lmplot(
        data=df[df['pes']==168], x='backbone size', y='cycles', hue='process'
    )
    plt.xlabel('Backbone size (number of layers)')
    plt.ylabel('Cycles')
    if 'VariableCNNBackbone' in config_dir:
        plt.title('Variable CNN Backbone Size vs. Cycles, 168 PEs')
    elif 'VariableBackbone' in config_dir:
        plt.title('FC Backbone Size vs. Cycles, 168 PEs')
    cycles_backbone.fig.savefig('cycles_backbone.png', bbox_inches="tight")
    # Cycles vs. performance
    cycles_performance = sns.lmplot(
        data=df[df['pes']==168], x='cycles', y='performance', hue='process'
    )
    plt.xlabel('Cycles')
    plt.ylabel('Epistemic uncertainty')
    if 'VariableCNNBackbone' in config_dir:
        plt.title('Variable CNN Cycles vs. Performance, 168 PEs')
    elif 'VariableBackbone' in config_dir:
        plt.title('FC Cycles vs. Performance, 168 PEs')
    cycles_performance.fig.savefig('cycles_performance.png', bbox_inches="tight")
    # Performance vs. backbone size
    performance_backbone = sns.lmplot(
        data=df[df['pes']==168], x='backbone size', y='performance'
    )
    plt.xlabel('Backbone size (number of layers)')
    plt.ylabel('Epistemic uncertainty')
    if 'VariableCNNBackbone' in config_dir:
        plt.title('Variable CNN Backbone Size vs. Performance, 168 PEs')
    elif 'VariableBackbone' in config_dir:
        plt.title('FC Backbone Size vs. Performance, 168 PEs')
    performance_backbone.fig.savefig('performance_backbone.png', bbox_inches="tight")
    """
    # Backbone size vs. # PEs vs. energy (heatmap) serial
    plt.clf()
    df_serial = df[df['process'] == 'Serial']
    print(df_serial)
    backbone_pes_energy = sns.heatmap(
        data=df_serial.pivot(index='pes', columns='backbone size', values='energy'),
        annot=True, fmt='.3f'
    )
    plt.xlabel('Backbone size (number of layers)')
    plt.ylabel('Number of PEs')
    if 'VariableCNNBackbone' in config_dir:
        plt.title('Serial Variable CNN Backbone Size vs. Number of PEs vs. Energy')
    elif 'VariableBackbone' in config_dir:
        plt.title('Serial FC Backbone Size vs. Number of PEs vs. Energy')
    plt.savefig('backbone_pes_energy_serial.png', bbox_inches="tight")
    # Backbone size vs. # PEs vs. energy (heatmap) serial
    plt.clf()
    df_serial = df[df['process'] == 'Serial']
    print(df_serial)
    backbone_pes_energy = sns.heatmap(
        data=df_serial.pivot(index='pes', columns='backbone size', values='energy per'),
        annot=True, fmt='.3f', annot_kws={'size': 7}
    )
    plt.xlabel('Backbone size (number of layers)')
    plt.ylabel('Number of PEs')
    if 'VariableCNNBackbone' in config_dir:
        plt.title('Serial Variable CNN Backbone Size vs. Number of PEs vs. Energy')
    elif 'VariableBackbone' in config_dir:
        plt.title('Serial FC Backbone Size vs. Number of PEs vs. Energy')
    plt.savefig('backbone_pes_energy_per_serial.png', bbox_inches="tight")
    # Backbone size vs. # PEs vs. energy (heatmap) serial
    plt.clf()
    df_serial = df[df['process'] == 'Serial']
    print(df_serial)
    backbone_pes_energy = sns.heatmap(
        data=df_serial.pivot(index='pes', columns='backbone size', values='ifmap spad'),
        annot=True, fmt='.3f', annot_kws={'size': 7}
    )
    plt.xlabel('Backbone size (number of layers)')
    plt.ylabel('Number of PEs')
    if 'VariableCNNBackbone' in config_dir:
        plt.title('Serial Variable CNN Backbone Size vs. Number of PEs vs. ifmap spad pJ/Compute')
    elif 'VariableBackbone' in config_dir:
        plt.title('Serial FC Backbone Size vs. Number of PEs vs. ifmap spad pJ/Compute')
    plt.savefig('backbone_pes_ifmap_serial.png', bbox_inches="tight")
    # Backbone size vs. # PEs vs. energy (heatmap) parallel
    plt.clf()
    df_parallel = df[df['process'] == 'Parallel']
    print(df_parallel)
    backbone_pes_energy = sns.heatmap(
        data=df_parallel.pivot(index='pes', columns='backbone size', values='energy'),
        annot=True, fmt='.3f'
    )
    plt.xlabel('Backbone size (number of layers)')
    plt.ylabel('Number of PEs')
    if 'VariableCNNBackbone' in config_dir:
        plt.title('Parallel Variable CNN Backbone Size vs. Number of PEs vs. Energy')
    elif 'VariableBackbone' in config_dir:
        plt.title('Parallel FC Backbone Size vs. Number of PEs vs. Energy')
    plt.savefig('backbone_pes_energy_parallel.png', bbox_inches="tight")
    # Backbone size vs. # PEs vs. energy (heatmap) parallel
    plt.clf()
    df_parallel = df[df['process'] == 'Parallel']
    print(df_parallel)
    backbone_pes_energy = sns.heatmap(
        data=df_parallel.pivot(index='pes', columns='backbone size', values='energy per'),
        annot=True, fmt='.3f', annot_kws={'size': 7}
    )
    plt.xlabel('Backbone size (number of layers)')
    plt.ylabel('Number of PEs')
    if 'VariableCNNBackbone' in config_dir:
        plt.title('Parallel Variable CNN Backbone Size vs. Number of PEs vs. Energy')
    elif 'VariableBackbone' in config_dir:
        plt.title('Parallel FC Backbone Size vs. Number of PEs vs. Energy')
    plt.savefig('backbone_pes_energy_per_parallel.png', bbox_inches="tight")
    # Backbone size vs. # PEs vs. energy (heatmap) parallel
    plt.clf()
    df_parallel = df[df['process'] == 'Parallel']
    print(df_parallel)
    backbone_pes_energy = sns.heatmap(
        data=df_parallel.pivot(index='pes', columns='backbone size', values='ifmap spad'),
        annot=True, fmt='.3f', annot_kws={'size': 7}
    )
    plt.xlabel('Backbone size (number of layers)')
    plt.ylabel('Number of PEs')
    if 'VariableCNNBackbone' in config_dir:
        plt.title('Parallel Variable CNN Backbone Size vs. Number of PEs vs. ifmap spad pJ/Compute')
    elif 'VariableBackbone' in config_dir:
        plt.title('Parallel FC Backbone Size vs. Number of PEs vs. ifmap spad pJ/Compute')
    plt.savefig('backbone_pes_ifmap_parallel.png', bbox_inches="tight")
    # Backbone size vs. # PEs vs. cycles (heatmap) serial
    plt.clf()
    df_serial = df[df['process'] == 'Serial']
    backbone_pes_cycles = sns.heatmap(
        data=df_serial.pivot(index='pes', columns='backbone size', values='cycles'),
        annot=True, fmt='.0f'
    )
    plt.xlabel('Backbone size (number of layers)')
    plt.ylabel('Number of PEs')
    if 'VariableCNNBackbone' in config_dir:
        plt.title('Serial Variable CNN Backbone Size vs. Number of PEs vs. Number of Cycles')
    elif 'VariableBackbone' in config_dir:
        plt.title('Serial FC Backbone Size vs. Number of PEs vs. Number of Cycles')
    plt.savefig('backbone_pes_cycles_serial.png', bbox_inches="tight")
    # Backbone size vs. # PEs vs. energy (heatmap) parallel
    plt.clf()
    df_parallel = df[df['process'] == 'Parallel']
    backbone_pes_cycles = sns.heatmap(
        data=df_parallel.pivot(index='pes', columns='backbone size', values='cycles'),
        annot=True, fmt='.0f'
    )
    plt.xlabel('Backbone size (number of layers)')
    plt.ylabel('Number of PEs')
    if 'VariableCNNBackbone' in config_dir:
        plt.title('Parallel Variable CNN Backbone Size vs. Number of PEs vs. Number of Cycles')
    elif 'VariableBackbone' in config_dir:
        plt.title('Parallel FC Backbone Size vs. Number of PEs vs. Number of Cycles')
    plt.savefig('backbone_pes_cycles_parallel.png', bbox_inches="tight")
    """
    return results


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
    parser.add_argument('--constraints', type=str, default=None, help='Include configs with this str')
    parser.add_argument('--base_dir', type=str, help='Base directory')
    parser.add_argument('--top_dir', type=str, default="layer_shapes", help="Directory with layer shapes")
    parser.add_argument('--config_dir', type=str, default="configs", help="Directory with configs")
    parser.add_argument('--performance_csv', type=str, default="performance_results", help="CSV file of performance")
    parser.add_argument('--design', type=str, default="eyeriss_like", help="Architecture design")
    parser.add_argument('--model_type', type=str, default="VariableBackbone", help="Name of model")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_options()
    #aggregate(args)
    plot(args, args.config_dir, args.performance_csv)
