#!/bin/bash

SEEDS=(42 43 44 45 46)


for seed in "${SEEDS[@]}"; do
    sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=0:50:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -s $seed -f False -nl 2 -dr 0 -la 0 -lr 0.01 -it False -bs 128 -hd 256 -nsl 3 -bn False -re False -ll 7 -al 4 -c False"
    sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=0:50:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -s $seed -f False -nl 2 -dr 0 -la 0.1 -lr 0.01 -it True -bs 128 -hd 256 -nsl 3 -bn False -re False -ll 7 -al 4 -c True"
    sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=0:50:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m True -s $seed -f False -nl 2 -dr 0 -la 0.1 -lr 0.01 -it False -bs 128 -hd 256 -nsl 3 -bn False -re False -ll 7 -al 4 -c True"
    sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=0:50:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -s $seed -f False -nl 2 -dr 0 -la 0.1 -lr 0.01 -it False -bs 128 -hd 256 -nsl 3 -bn False -re False -ll 7 -al 4 -c True"
done
