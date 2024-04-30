#!/bin/bash
#SBATCH -J PNULearning
#SBATCH -o ./logs/%j.out
#SBATCH -a 0-9

UNLABELLED_RATE=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

python ./src/main.py --output_file ./output/PNU_rate.csv --unlabel_rate ${UNLABELLED_RATE[$SLURM_ARRAY_TASK_ID]}
