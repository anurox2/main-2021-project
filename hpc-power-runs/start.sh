#!/bin/bash

echo "Reached before python lines"
module load Python/3.7.4-GCCcore-8.3.0
source /home/x397j446/main-project/env1/bin/activate
echo "$(python -V 2>&1)" > /home/x397j446/file10.out
export PYTHONDONTWRITEBYTECODE=1
python /home/x397j446/main-project/autoencoder_options.py
echo "After python lines"
