#!/usr/bin/env bash

python3 -m profiler --params=VariableCNNBackbone/16--1-32--1-64-128-fc512-fc2shape_0split_5heads_serial --model_type=VariableCNNBackbone --design=eyeriss_like_84pe
python3 -m profiler --params=VariableCNNBackbone/16--1-32--1-64-128-fc512-fc2shape_1split_5heads_serial --model_type=VariableCNNBackbone --design=eyeriss_like_84pe
python3 -m profiler --params=VariableCNNBackbone/16--1-32--1-64-128-fc512-fc2shape_2split_5heads_serial --model_type=VariableCNNBackbone --design=eyeriss_like_84pe
python3 -m profiler --params=VariableCNNBackbone/16--1-32--1-64-128-fc512-fc2shape_3split_5heads_serial --model_type=VariableCNNBackbone --design=eyeriss_like_84pe
python3 -m profiler --params=VariableCNNBackbone/16--1-32--1-64-128-fc512-fc2shape_4split_5heads_serial --model_type=VariableCNNBackbone --design=eyeriss_like_84pe
python3 -m profiler --params=VariableCNNBackbone/16--1-32--1-64-128-fc512-fc2shape_5split_5heads_serial --model_type=VariableCNNBackbone --design=eyeriss_like_84pe
