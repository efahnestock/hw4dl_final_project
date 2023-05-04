#!/usr/bin/env bash

python3 -m profiler --params=VariableBackbone/1-20-30-40shape_0split_3heads_serial --model_type=VariableBackbone --design=eyeriss_like
python3 -m profiler --params=VariableBackbone/1-20-30-40shape_1split_3heads_serial --model_type=VariableBackbone --design=eyeriss_like
python3 -m profiler --params=VariableBackbone/1-20-30-40shape_2split_3heads_serial --model_type=VariableBackbone --design=eyeriss_like
