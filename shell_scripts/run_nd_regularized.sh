#!/bin/bash

SEEDS=(42 43 44 45 46)


for seed in "${SEEDS[@]}"; do
    sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=0:50:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -s $seed -f False -nl 3 -dr 0.5 -la 0.1 -lr 0.01 -it False -bs 128 -hd 128 -nsl 1 -bn True -re True -ll 7 -al 4 -c True"
done

