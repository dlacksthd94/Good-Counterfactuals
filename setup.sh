#!/bin/bash
set -e

RESET_ENV=False
ENV_NAME="260"

source ~/miniconda3/etc/profile.d/conda.sh

if conda info --envs | awk '{print $1}' | grep -q "^$ENV_NAME$"; then
    if [ "$RESET_ENV" = "True" ]; then
        conda deactivate
        echo "Resetting conda environment '$ENV_NAME'..."
        conda env remove -y -n "$ENV_NAME"

        echo "Creating conda environment '$ENV_NAME'..."
        conda create -y -n "$ENV_NAME" python=3.10
    else
        echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
    fi
else
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -y -n "$ENV_NAME" python=3.10
fi

conda activate "$ENV_NAME"

# pip cache purge
pip install -r requirements.txt

echo "\nSUCCESSFULLY DONE!"