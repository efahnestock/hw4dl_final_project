# BONE: Backbone Optimization for Neural Ensembles

## Setting up the python package
Run this from the top level directory (with `setup.py` in it)
```pip install -e .```

## Training linear experiments 
To run the fully connected experiments with the default parameters, run the following command from the top level directory:
```mkdir experiments && python hw4dl/tools/run_fc_experiment.py```

# Running hardware 

## Convert PyTorch to Timeloop 

Create and activate the Conda environment:

```
conda env create --name hw4dl --file conda_mac.yaml
conda activate hw4dl
```

Run the scripts in `scripts/convert`. For example, to convert all of the VariableBackbone variants in parallel, run:

```
cd workspace/final-project/
bash ./scripts/convert/VariableBackbone_parallel.sh
```

This will populate the `layer_shapes` directory with the converted Timeloop problem for each layer.

## Run TimeLoop and Accelergy

Please pull the docker first to update the container, and then start with `docker-compose up`. 

```
export DOCKER_ARCH=amd64
docker-compose pull
docker-compose up
```

You may need to run `pip install tqdm`.

Run the scripts in `scripts/profile`. For example, to run all of the VariableBackbone variants in parallel on Eyeriss architecture with 42 PEs, run:

```
cd workspace/final-project/
bash ./scripts/profile/VariableBackbone_parallel_42.sh
```

This will populate `timeloop_results`. In the example above, the results will be in `timeloop_results/eyeriss_like_42pe/VariableBackbone`.

## Visualize results

Put a CSV with the epistemic uncertainty scores in a directory called `performance_results`. Then, run the scripts in `scripts/aggregate`.
