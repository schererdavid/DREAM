#!/bin/bash


SEEDS=(42 43 44 45 46)
TW=(1 3)
for seed in "${SEEDS[@]}"; do
    for tw in "${TW[@]}"; do
        sbatch -c 4 --mem-per-cpu=32G --time=8:00:00 --wrap "python3 main.py -t traditional_train_test_evaluate -d mimic -l [$tw] -tp random -ns 1 -s $seed -ab False -m False"
    done
done