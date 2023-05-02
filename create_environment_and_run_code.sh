#!/bin/bash

if [ $# -ne 4 ]; then
    echo "Error: This script requires 4 arguments"
    exit 1
fi

arg1=$1
arg2=$2
arg3=$3
arg4=$4
# Create virtual environment
python3.8 -m venv testEnv
# Activate virtual environment
source testEnv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create results folder
mkdir -p results
mkdir -p data

# Run main.py
python main.py "$arg1" "$arg2" "$arg3" "$arg4"

# Deactivate virtual environment
deactivate
