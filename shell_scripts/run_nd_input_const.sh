#!/bin/bash

SEEDS=(42 43 44 45 46)


for seed in "${SEEDS[@]}"; do
    #sbatch -c 4 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t construct_lstm -d mimic -ll 7 -al 4 -s $seed -ab False -m False"
    #sbatch -c 4 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t construct_lstm -d mimic -ll 7 -al 4 -s $seed -ab False -m True"
    sbatch -c 4 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t construct_lstm -d eicu -ll 7 -al 4 -s $seed -ab False -m False"
    sbatch -c 4 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t construct_lstm -d pic -ll 7 -al 4 -s $seed -ab False -m False"
done


#LOOKBACK=(2 7 14)
#for seed in "${SEEDS[@]}"; do
#    for lookback in "${LOOKBACK[@]}"; do
#        sbatch -c 4 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t construct_lstm -d mimic -ll $lookback -al 4 -s $seed -ab False -m False"
#     done
# done


# AGGREGATION=(1 4 8)
# for seed in "${SEEDS[@]}"; do
#     for aggregation in "${AGGREGATION[@]}"; do
#         sbatch -c 4 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t construct_lstm -d mimic -ll 7 -al $aggregation -s $seed -ab False -m False"
#     done
# done
