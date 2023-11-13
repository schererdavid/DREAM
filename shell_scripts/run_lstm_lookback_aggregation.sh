#!/bin/bash

SEEDS=(42 43 44 45 46)


AGGREGATION=(1 4 8)

LOOKBACK=(2 7 14)

for seed in "${SEEDS[@]}"; do
    for lookback in "${LOOKBACK[@]}"; do
        sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -s $seed -f False -nl 2 -dr 0 -la 0 -lr 0.01 -it False -bs 128 -hd 256 -nsl 3 -bn False -re False -ll $lookback -al 4"
    done
done

for seed in "${SEEDS[@]}"; do
    for aggregation in "${AGGREGATION[@]}"; do
        sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -s $seed -f False -nl 2 -dr 0 -la 0 -lr 0.01 -it False -bs 128 -hd 256 -nsl 3 -bn False -re False -ll 7 -al $aggregation"
    done
done


