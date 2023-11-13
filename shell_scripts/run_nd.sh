#!/bin/bash

LIN=(1 2)
DROPOUT=(0.4 0.5)
BN=(True False)
RELU=(True False)
LR=(0.01)
LA=(0.1 0.2 0.09)
BS=(256)
HD=(128)
NSL=(4 5)


for nsl in "${NSL[@]}"; do
    for hd in "${HD[@]}"; do
        for bs in "${BS[@]}"; do
            for la in "${LA[@]}"; do
                for lr in "${LR[@]}"; do
                    for dr in "${DROPOUT[@]}"; do
                        for lin in "${LIN[@]}"; do
                            for bn in "${BN[@]}"; do
                                for relu in "${RELU[@]}"; do
                                    sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=0:39:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -la $la -s 42 -f False -nl $lin -dr $dr -lr $lr -it False -bs $bs -hd $hd -nsl $nsl -bn $bn -re $relu"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

