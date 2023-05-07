#!/usr/bin/env bash

python3 -m aggregate --params=VariableBackbone/1-30-30-30-30-30-2shape_0split_5heads_parallel --base_dir=./ --design=eyeriss_like_84pe
python3 -m aggregate --params=VariableBackbone/1-30-30-30-30-30-2shape_1split_5heads_parallel --base_dir=./ --design=eyeriss_like_84pe
python3 -m aggregate --params=VariableBackbone/1-30-30-30-30-30-2shape_2split_5heads_parallel --base_dir=./ --design=eyeriss_like_84pe
python3 -m aggregate --params=VariableBackbone/1-30-30-30-30-30-2shape_3split_5heads_parallel --base_dir=./ --design=eyeriss_like_84pe
python3 -m aggregate --params=VariableBackbone/1-30-30-30-30-30-2shape_4split_5heads_parallel --base_dir=./ --design=eyeriss_like_84pe
python3 -m aggregate --params=VariableBackbone/1-30-30-30-30-30-2shape_5split_5heads_parallel --base_dir=./ --design=eyeriss_like_84pe
