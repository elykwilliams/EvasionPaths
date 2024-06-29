#!/bin/bash

## This is a sample script that can be used to run a given simulation

# activate environment (use weird hack, see https://github.com/conda/conda/issues/7980)
eval "$(conda shell.bash hook)"
conda activate EvasionPaths-env

# nohup will continue to run remotely after logging out
python3 sample_experiment.py

# deactivate environment
conda deactivate