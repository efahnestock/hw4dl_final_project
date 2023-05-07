#!/usr/bin/env bash

python3 -m aggregate --config_dir=./configs/VariableCNNBackbone --base_dir=./ --performance_csv=./performance_results/cnn_results.csv --model_type=VariableCNNBackbone --design=eyeriss_like_168pe
