#!/bin/bash

SEEDS=(42 43 44 45 46)

for seed in "${SEEDS[@]}"; do
    sbatch -c 4 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t construct_traditional -d mimic -l [2] -tp [0,1,2,3,4] -ns 1 -s $seed -ab False -m False"
done


