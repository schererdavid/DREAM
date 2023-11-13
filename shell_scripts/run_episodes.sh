#!/bin/bash
sbatch -c 4 --mem-per-cpu=32G --time=4:00:00 --wrap "python3 main.py -t episodes -d mimic -m True"
sbatch -c 4 --mem-per-cpu=32G --time=4:00:00 --wrap "python3 main.py -t episodes -d mimic -m False"
sbatch -c 4 --mem-per-cpu=32G --time=4:00:00 --wrap "python3 main.py -t episodes -d eicu -m False"
sbatch -c 4 --mem-per-cpu=32G --time=4:00:00 --wrap "python3 main.py -t episodes -d pic -m False"
