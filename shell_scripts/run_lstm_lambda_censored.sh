#!/bin/bash

SEEDS=(42 43 44 45 46)

LA_VALUES=(0 0.1 0.01 0.001 0.0001 0.00001 0.2 0.3 0.4 0.5)

for seed in "${SEEDS[@]}"; do
    for la in "${LA_VALUES[@]}"; do
        sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=0:40:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -s $seed -f False -nl 2 -dr 0 -la $la -lr 0.01 -it False -bs 128 -hd 256 -nsl 3 -bn False -re False -ll 7 -al 4 -c True"
    done
done


#for seed in "${SEEDS[@]}"; do
#    sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=0:40:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -s $seed -f False -nl 2 -dr 0 -la 0 -lr 0.01 -it False -bs 128 -hd 256 -nsl 3 -bn False -re False -ll 7 -al 4 -c False"
    #sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=0:40:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -s $seed -f False -nl 2 -dr 0 -la 0 -lr 0.01 -it False -bs 128 -hd 256 -nsl 3 -bn False -re False -ll 7 -al 4 -c True"
#done


