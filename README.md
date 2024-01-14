# Efficient Bellman ford Algorithm

![Bellman-Ford-Algorithm-3](https://github.com/Kilian36/bellman-ford-parallel/assets/94917153/272fff90-de43-4993-812d-203b5092840a)

This repository contains the code to run the Bellman-Ford algorithm in serveral parallel versions.

## REQUIREMENTS
make to use the prepared compilations and experiments.
gcc to run the openMP versions.
nvcc to run CUDA.

All the code is been tested only on linux machines.

## MAKEFILE
The makefile contains the targets to run the experiments reported in the pdf. Moreover it provides utilities to:
    - Generate graphs.
    - Clean folders and executables.
    - Run with different settings.

If you don't work with a SLURM MACHINE you can run the three experiments by calling
    - make run_exp1
    - make run_exp2  # Slow, may take half an hour
    - make run_exp3

Otherwise you must write a sbatch script run_script.sbatch, which generally as the following structure:
```
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
```
You should write this directly in a linux machine to avoid problems with ```\n``` characters.

All the parameters in the makefile can be changed to perform some new experiments. 

## CODE STRUCTURE
Every C/CU source code file has its own main. It works as follows:
- It reads graphs from files in the path ./tests/graphs/ 
  The graph file structure is:  
   $n_i$   $n_j$   $w_{ij}$   
   $n_k$   $n_l$   $w_{kl}$  
- It computes the bellman ford algorithm and stores all the distances and times (if != form baseline)
  at the path ./results/model_name/distances and ./results/model_name/times.txt. If you run the baseline, 
  it solves the graph and put the solution in grountruths. 
      
The graph generator is written in python and puts all the graphs at the needed path.

The model checker compares the output of all the distances folders in results with the ones in grountruths.

For a more detailed structure of the implementations refer to the project report found in this report. 


