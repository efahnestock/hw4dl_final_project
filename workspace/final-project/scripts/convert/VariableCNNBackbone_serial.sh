#!/usr/bin/env bash

python3 -m convert --params=VariableCNNBackbone/_0split_5heads_serial --model_type=VariableCNNBackbone
python3 -m convert --params=VariableCNNBackbone/_1split_5heads_serial --model_type=VariableCNNBackbone
python3 -m convert --params=VariableCNNBackbone/_2split_5heads_serial --model_type=VariableCNNBackbone
python3 -m convert --params=VariableCNNBackbone/_3split_5heads_serial --model_type=VariableCNNBackbone
python3 -m convert --params=VariableCNNBackbone/_4split_5heads_serial --model_type=VariableCNNBackbone
python3 -m convert --params=VariableCNNBackbone/_5split_5heads_serial --model_type=VariableCNNBackbone
