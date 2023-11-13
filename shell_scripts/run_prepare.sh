#!/bin/bash

sbatch -c 4 --mem-per-cpu=32G --time=4:00:00 --wrap "python3 main.py -t prepare -d mimic"
sbatch -c 4 --mem-per-cpu=32G --time=4:00:00 --wrap "python3 main.py -t prepare -d eicu"
sbatch -c 4 --mem-per-cpu=32G --time=4:00:00 --wrap "python3 main.py -t prepare -d pic"
