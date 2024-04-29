#!/bin/bash
#SBATCH -J PNULearning
#SBATCH -o ./logs/%j.out

python ./src/main.py