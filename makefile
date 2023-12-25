# Examples of runs, modify the SIZE parameters as you wish. 
# 1. make clean_results  				   --> Deletes all the saved results for all the models.
# 2. make clean_results all run_omp clean  --> Run and save results for all threads settings and sizes. 
# 3. make clean_results all run_cuda clean --> Run and save results for all threads settings. (500 and 1000 nodes).
# 4. make all run_base clean			   --> Generates the groundtruths for all the grpahs generated (standard size and 1000).
# 4. make check							   --> Run the model checker

# If you want to try bigger graphs ensure that the run commands are updated or dirctly compile and run the singular files. 

# SIZE OF THE GRAPHS
STANDARD := \
	 500 \

BIG := \
	1000 

# OMP PARAMETERS
THREAD_SETTINGS_OMP := \
	1 \
	2 \
	4 \
	8

MODELS_OMP := \
	"locks" \
	"simple" \
	"frontier"


# CUDA PARAMETERS
THREAD_SETTINGS_CUDA := \
	100 \
	250 \
	500 \

MODELS_CUDA := \
	"simple" \
	"frontier"

# Clean the test folder if any, useful to keep the repo lightwiegth
clean_tests:
	rm -rf tests

# Clean the results folder if any, use always before running the models
clean_results:
	rm -rf results 

# It removes all the executables
clean_prog:
	rm -rf *.o prog \

# Compile all the source files and generates an executable for each of them
# Ensure to run this command with the nvcc compiler installed.
all_omp:
	gcc -o bf_frontier.o functions.c bf-omp-frontier.c -fopenmp -lm; \
	gcc -o bf_locks.o functions.c bf-omp-locks.c -fopenmp -lm; \
	gcc -o bf_simple.o functions.c bf-omp-simple.c -fopenmp -lm; \
	gcc -o base.o functions.c bf-baseline.c -lm; \

all_cuda:
	nvcc -o cuda_frontier.o functions.c bf-cuda-frontier.cu -lm; \
	nvcc -o cuda_simple.o functions.c bf-cuda-simple.cu -lm; \
	
# Runs all the OMP models using different thread numbers
# and different sizes of the graphs. Big graphs are excluded
# for time reasons
run_omp:
	@for model in $(MODELS_OMP); do \
		MODEL="./bf_$${model}.o"; \
		for thread in $(THREAD_SETTINGS_OMP); do \
			for param in $(STANDARD); do \
				set -- $$param; \
				echo "    GRAPHS   : 25           "; \
				echo "    SIZE     : $$1          "; \
				echo "    MODEL IS : $${model}    "; \
				echo "    THREADS  : $$thread     "; \
				OMP_NUM_THREADS=$$thread $$MODEL 25 $$1  ; \
				echo "    RUN ENDED"; \
			done; \
		python3 ./utils/src/model_checker.py -o "results/omp-$${model}/distances" -g "tests/groundtruths" -v 0; \
		done; \
	done;

# Runs the baseline model and generate the groundtruths. 
run_base:
	@for param in $(STANDARD); do \
		echo "    GRAPHS   : 25           "; \
		echo "    SIZE     : $$param          "; \
		echo "    MODEL    : baseline    "; \
		set  -- $$param; \
		./base.o 25 $$param; \
		echo "RUN ENDED"; \
	done; \
	for param in $(BIG); do \
		echo "    GRAPHS   : 5           "; \
		echo "    SIZE     : $$param          "; \
		echo "    MODEL    : baseline    "; \
		set  -- $$param; \
		./base.o 5 $$param; \
		echo "RUN ENDED"; \
	done; \
	
# Runs all the CUDA models using different thread numbers
# and different sizes of the graphs. Only the 500 and 1000
# nodes are ran here. 
run_cuda:
	@for model in $(MODELS_CUDA); do \
		MODEL="./cuda_$${model}.o"; \
		for thread in $(THREAD_SETTINGS_CUDA); do \
			echo "    GRAPHS   : 25           "; \
			echo "    SIZE     : 500         "; \
			echo "    MODEL IS : $${model}    "; \
			echo "    THREADS  : $$thread     "; \
			set -- $$thread; \
			$$MODEL 25 500 $$1; \
			\
			echo "    GRAPHS   : 5           "; \
			echo "    SIZE     : 1000          "; \
			echo "    MODEL IS : $${model}    "; \
			echo "    THREADS  : $$thread     "; \
			set -- $$thread; \
			$$MODEL 5 1000 $$1; \
			python3 ./utils/src/model_checker.py -o "results/cuda-$${model}/distances" -g "tests/groundtruths"; \
		done; \
	done; \

# Generates a certain number of graphs 25 here of all the sizes specified in PARAMETERS
# The parameters can be changed looking at ./utils/src/generate_graphs.py
gen_graphs:
	@for param in $(STANDARD); do \
		python3 ./utils/src/generate_graphs.py -n 25 -s $${param} --negative_edges 0.0; \
	done; \
	for param in $(BIG); do \
		python3 ./utils/src/generate_graphs.py -n 5 -s $${param} --negative_edges 0.0; \
	done; 

# For all cuda and omp model specified check the distances saved. If you run after run_cuda or
# run_omp it will check the result with the higher number of threads.
check:
	@for model in $(MODELS_OMP); do \
		echo "CHECKING RESULTS FOR MODEL: $${model}\n"; \
		python3 ./utils/src/model_checker.py -o "results/omp-$${model}/distances" -g "tests/groundtruths"; \
		echo "\n"; \
	done; \

	@for model in $(MODELS_CUDA); do \
		echo "CHECKING RESULTS FOR MODEL: $${model}\n"; \
		python3 ./utils/src/model_checker.py -o "results/cuda-$${model}/distances" -g "tests/groundtruths"; \
		echo "\n"; \
	done; \


