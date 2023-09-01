#! /bin/bash

# Script builds/activates 3d_pipeline conda environment
# Note: this script should be sourced eg. `. build_environment.sh`

ENV_NAME="3d_pipeline"

# Check if conda is installed
if ! command -v conda &>/dev/null; then
    echo "Anaconda (or Miniconda) is not installed. Please install or load it before running this script."
    exit 1
fi

# Activate error handling
set -e

# Check if the environment already exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Activating existing Anaconda environment: $ENV_NAME"
    conda activate "$ENV_NAME"
else
    echo "Creating and activating a new Anaconda environment: $ENV_NAME"
    conda create -y -n "$ENV_NAME" python=3.9
    conda activate "$ENV_NAME"
    pip install -r requirements.txt
    echo "Environment ready"
fi

set +e
