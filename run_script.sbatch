#!/bin/bash
#SBATCH --job-name="bellman-ford-parallel"
#SBATCH --mail-type=ALL
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --output=terminal_file.txt
#SBATCH --gres=gpu:1
#SBATCH --cores-per-socket=4

echo "EXP 1"
make run_exp1
echo "DONE"
echo "-----------------------------------------------------------------------------------------------------" 
echo "EXP 2"
make run_exp2
echo "DONE"
echo "-----------------------------------------------------------------------------------------------------"
echo "EXP 3"
make run_exp3
echo "DONE"
echo "-----------------------------------------------------------------------------------------------------"

echo "-----------------------------------------------------------------------------------------------------"
lscpu
make clean_prog


