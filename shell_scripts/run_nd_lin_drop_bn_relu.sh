#!/bin/bash

LIN=(0 1 2)
DROPOUT=(0 0.3 0.5)
BN=(True False)
RELU=(True False)
for dr in "${DROPOUT[@]}"; do
    for lin in "${LIN[@]}"; do
        for bn in "${BN[@]}"; do
            for relu in "${RELU[@]}"; do
                sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=4:00:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -la 0 -s 42 -f False -nl $lin -dr $dr -lr 0.01 -it False -bs 128 -hd 256 -nsl 3 -bn $bn -re $relu"
            done
        done
    done
done

#LR=(0 0.1 0.2 0.3 0.4)
#for lr in "${LR[@]}"; do
#    sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=4:00:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -la 0 -s 42 -f False -nl $lin -dr $dr -lr 0.01 -it False -bs 128 -hd 256 -nsl 3 -bn $bn -re $relu"
#done


