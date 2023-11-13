#!/bin/bash

# Array der gewünschten Werte für -la
LA_VALUES=(0 0.1 0.01 0.001 0.0001)

# Schleife durch jeden Wert in LA_VALUES
for la in "${LA_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=4:00:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -s $seed -f False -nl 0 -dr 0.3 -la $la -lr 0.01 -it False -bs 128 -hd 256 -nsl 3"
    done
done
