#!/bin/bash
# Script to run Python with TF_DISABLE_PLUGIN_LOADING=1

# Set the environment variable
export TF_DISABLE_PLUGIN_LOADING=1

# Run the Python script with the specified Python interpreter
/opt/miniconda3/envs/dl-env/bin/python "$@"
