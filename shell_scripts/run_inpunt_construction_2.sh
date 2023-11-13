#!/bin/bash

SEEDS=(42 43 44 45 46)

for seed in "${SEEDS[@]}"; do
    #sbatch -c 4 --mem-per-cpu=32G --time=6:00:00 --wrap "python3 main.py -t input -d mimic -l [2] -tp random -ns 1 -s $seed -ab False -m False"
    sbatch -c 4 --mem-per-cpu=32G --time=6:00:00 --wrap "python3 main.py -t input -d mimic -l [2] -tp random -ns 2 -s $seed -ab False -m False"
    sbatch -c 4 --mem-per-cpu=32G --time=6:00:00 --wrap "python3 main.py -t input -d mimic -l [2] -tp random -ns 3 -s $seed -ab False -m False"
done

for seed in "${SEEDS[@]}"; do
    #sbatch -c 4 --mem-per-cpu=32G --time=6:00:00 --wrap "python3 main.py -t input -d mimic -l [2] -tp random -ns 1 -s $seed -ab False -m False"
    sbatch -c 4 --mem-per-cpu=32G --time=6:00:00 --wrap "python3 main.py -t input -d mimic -l [1] -tp random -ns 1 -s $seed -ab False -m False"
    sbatch -c 4 --mem-per-cpu=32G --time=6:00:00 --wrap "python3 main.py -t input -d mimic -l [1] -tp random -ns 1 -s $seed -ab False -m False"
    sbatch -c 4 --mem-per-cpu=32G --time=6:00:00 --wrap "python3 main.py -t input -d mimic -l [3] -tp random -ns 1 -s $seed -ab False -m False"
    sbatch -c 4 --mem-per-cpu=32G --time=6:00:00 --wrap "python3 main.py -t input -d mimic -l [3] -tp random -ns 1 -s $seed -ab False -m False"
done


