#!/bin/bash



LR=(0.01 0.1 0.001 0.0001)
for lr in "${LR[@]}"; do
    sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -la 0 -s 42 -f False -nl 0 -dr 0 -lr $lr -it False -bs 128 -hd 256 -nsl 3 -bn False -re False"
done

LA=(0 0.1 0.01 0.001 0.0001 0.00001)
for la in "${LA[@]}"; do
    sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -la $la -s 42 -f False -nl 0 -dr 0 -lr 0.01 -it False -bs 128 -hd 256 -nsl 3 -bn False -re False"
done

BS=(128 64 32 256)
for bs in "${BS[@]}"; do
    sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -la 0 -s 42 -f False -nl 0 -dr 0 -lr 0.01 -it False -bs $bs -hd 256 -nsl 3 -bn False -re False"
done

HD=(256 128 512 64)
for hd in "${HD[@]}"; do
    sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -la 0 -s 42 -f False -nl 0 -dr 0 -lr 0.01 -it False -bs 128 -hd $hd -nsl 3 -bn False -re False"
done

NSL=(1 2 3 4 5)
for nsl in "${NSL[@]}"; do
    sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab False -m False -la 0 -s 42 -f False -nl 0 -dr 0 -lr 0.01 -it False -bs 128 -hd 256 -nsl $nsl -bn False -re False"
done

sbatch -c 4 -p gpu --gres=gpu:1 --mem-per-cpu=32G --time=1:00:00 --wrap "python3 main.py -t lstm_train_test_evaluate -d mimic -ab True -m False -la 0 -s 42 -f False -nl 0 -dr 0 -lr 0.01 -it False -bs 128 -hd 256 -nsl 3 -bn False -re False"