#!/usr/bin/env bash

python3 -m profiler --params=VariableCNNBackbone/1-16--1-32--1-64-fc512-fc18shape_0split_5heads_parallel --model_type=VariableCNNBackbone --design=eyeriss_like_672pe
python3 -m profiler --params=VariableCNNBackbone/1-16--1-32--1-64-fc512-fc18shape_1split_5heads_parallel --model_type=VariableCNNBackbone --design=eyeriss_like_672pe
python3 -m profiler --params=VariableCNNBackbone/1-16--1-32--1-64-fc512-fc18shape_2split_5heads_parallel --model_type=VariableCNNBackbone --design=eyeriss_like_672pe
python3 -m profiler --params=VariableCNNBackbone/1-16--1-32--1-64-fc512-fc18shape_3split_5heads_parallel --model_type=VariableCNNBackbone --design=eyeriss_like_672pe
python3 -m profiler --params=VariableCNNBackbone/1-16--1-32--1-64-fc512-fc18shape_4split_5heads_parallel --model_type=VariableCNNBackbone --design=eyeriss_like_672pe
python3 -m profiler --params=VariableCNNBackbone/1-16--1-32--1-64-fc512-fc18shape_5split_5heads_parallel --model_type=VariableCNNBackbone --design=eyeriss_like_672pe

