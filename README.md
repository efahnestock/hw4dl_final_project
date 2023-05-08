# BONE: Backbone Optimization for Neural Ensembles

## Setting up the python package
Run this from the top level directory (with `setup.py` in it)
```pip install -e .```

## Training Linear Experiments 
To run the fully connected experiments with the default parameters, run the following command from the top level directory:
```mkdir experiments && python hw4dl/tools/run_fc_experiment.py```

# Running hardware 
For detailed instructions, please read [./workspace/final-project/README.md](./workspace/final-project/README.md). 

## Convert PyTorch to Timeloop 

## Run TimeLoop and Accelergy

Please pull the docker first to update the container, and then start with `docker-compose up`. 
```
git clone https://github.com/aknh9189/hw4dl_final_project.git
docker-compose pull
docker-compose up
```
Note that we use the [eyeriss-like design](https://people.csail.mit.edu/emer/papers/2017.01.jssc.eyeriss_design.pdf)

