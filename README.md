This repository contains the code to run the Bellman-Ford algorithm in serveral parallel versions.

## REQUIREMENTS
make to use the prepared compilations and experiments.
gcc to run the openMP versions.
nvcc to run CUDA.

All the code is been tested only on linux machines.

## MAKEFILE
The makefile contains the targets to run the experiments reported in the pdf.

If you don't work with a SLURM MACHINE you can run the three experiments by calling

make run_exp1
make run_exp2  # Slow, make take half an hour
make run_exp3

All the parameters in the makefile can be changed to perform some new experiments. 


## CODE STRUCTURE
Every C/CU source code file has its own main. It works as follows:
    - It reads graphs from files in the path ./tests/graphs/ 
      The graph file structure is:
       ni nj wij
       nk nl wkl
    - It computes the bellman ford algorithm and stores all the distances and times (if != form baseline)
      at the path ./results/model_name/distances and ./results/model_name/times.txt. If you run the baseline, 
      it solves the graph and put the solution in grountruths. 

The graph generator is written in python and puts all the graphs at the needed path.

The model checker compares the output of all the distances folders in results with the ones in grountruths.
 
