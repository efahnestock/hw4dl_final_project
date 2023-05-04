#!/usr/bin/env bash

python3 -m convert --params=VariableBackbone/1-20-30-40shape_0split_3heads_parallel --model_type=VariableBackbone
python3 -m convert --params=VariableBackbone/1-20-30-40shape_1split_3heads_parallel --model_type=VariableBackbone
python3 -m convert --params=VariableBackbone/1-20-30-40shape_2split_3heads_parallel --model_type=VariableBackbone
