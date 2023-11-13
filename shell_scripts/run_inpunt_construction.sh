#!/bin/bash

SEEDS=(42 43 44 45 46)

for seed in "${SEEDS[@]}"; do
    sbatch -c 4 --mem-per-cpu=32G --time=6:00:00 --wrap "python3 main.py -t input -d mimic -l [2] -tp random -ns 1 -s $seed -ab True -m True"
    sbatch -c 4 --mem-per-cpu=32G --time=6:00:00 --wrap "python3 main.py -t input -d mimic -l [2] -tp random -ns 1 -s $seed -ab False -m True"
    sbatch -c 4 --mem-per-cpu=32G --time=6:00:00 --wrap "python3 main.py -t input -d mimic -l [2] -tp random -ns 1 -s $seed -ab True -m False"
    sbatch -c 4 --mem-per-cpu=32G --time=6:00:00 --wrap "python3 main.py -t input -d mimic -l [2] -tp random -ns 1 -s $seed -ab False -m False"
    sbatch -c 4 --mem-per-cpu=32G --time=6:00:00 --wrap "python3 main.py -t input -d eicu -l [2] -tp random -ns 1 -s $seed -ab False -m False"
    sbatch -c 4 --mem-per-cpu=32G --time=6:00:00 --wrap "python3 main.py -t input -d pic -l [2] -tp random -ns 1 -s $seed -ab False -m False"
done



