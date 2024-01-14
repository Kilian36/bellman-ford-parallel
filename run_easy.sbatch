#!/bin/bash
#SBATCH --job-name="bellman-ford-parallel"
#SBATCH --mail-type=ALL
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --output=terminal_file.txt
#SBATCH --gres=gpu:1
#SBATCH --cores-per-socket=4

make clean_all gen_pos_graphs all run_base run_omp run_cuda clean_prog  # FASTER
#make clean_all gen_neg_graphs all run_base run_omp run_cuda clean_prog # SLOW HALF AN HOUR
lscpu
