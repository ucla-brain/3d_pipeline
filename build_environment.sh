#! /bin/bash

# Script builds/activates 3d_pipeline conda environment
# Note: this script should be sourced eg. `. build_environment.sh`

ENV_NAME="3d_pipeline"

# checks if previous command finished successfully
if_error_echo() {
    if [ $? -ne 0 ]; then
        # echo message if provided
        echo $1
        exit 61
    fi
}

# Check if conda is installed
if ! command -v conda &>/dev/null; then
    echo "Anaconda (or Miniconda) is not installed. Please install or load it before running this script."
    exit 1
fi

# Check if the environment already exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Activating existing Anaconda environment: $ENV_NAME"
    conda activate "$ENV_NAME"
else
    echo "Creating and activating a new Anaconda environment: $ENV_NAME"
    conda create -y -n "$ENV_NAME" python=3.9
    if_error_echo "Problem creating Anaconda environment"
    conda activate "$ENV_NAME"
    if_error_echo "Could not activate environment, this file must be sourced"
    pip install -r requirements.txt
    if_error_echo "Problem installing pip requirements"
    echo "Environment ready"
fi
