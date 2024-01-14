# SIZE OF THE GRAPHS
SIZE := \
	 500 \
	 1000 \

NUM_GRAPHS=10

# OMP PARAMETERS
THREAD_SETTINGS_OMP := \
	1 \
	2 \
	4 \
	8

MODELS_OMP := \
	"simple" \
	"frontier" \
	"locks" \

# CUDA PARAMETERS
THREAD_SETTINGS_CUDA := \
	512

MODELS_CUDA := \
	"simple" \
	"frontier" \

#											CLEANING TARGETS
################################################################################################################
# Clean the test folder if any, useful to keep the repo lightwiegth
clean_tests:
	rm -rf tests

# Clean the results folder if any, use always before running the models
clean_results:
	rm -rf results

# It removes all the executables
clean_prog:
	rm -rf *.o prog \

clean_all: clean_results clean_prog clean_tests
################################################################################################################


#											COMPILE TARGETS
################################################################################################################
# Compile all the source files and generates an executable for each of them
# Ensure to run this command with the nvcc compiler installed.
omp:
	gcc -o bf_frontier.o functions.c bf-omp-frontier.c -fopenmp -lm; \
	gcc -o bf_locks.o functions.c bf-omp-locks.c -fopenmp -lm; \
	gcc -o bf_simple.o functions.c bf-omp-simple.c -fopenmp -lm; 

base:
	gcc -o base.o functions.c bf-baseline.c -lm; 

cuda:
	nvcc -o cuda_frontier.o functions.c bf-cuda-frontier.cu -lm; \
	nvcc -o cuda_simple.o functions.c bf-cuda-simple.cu -lm; \

all: omp cuda base 
################################################################################################################


#											RUN TARGETS
################################################################################################################
# Runs all the OMP models using different thread numbers
# and different sizes of the graphs. Big graphs are excluded
# for time reasons
run_omp:
	@for model in $(MODELS_OMP); do \
		MODEL="./bf_$${model}.o"; \
		for thread in $(THREAD_SETTINGS_OMP); do \
			for param in $(SIZE); do \
				set -- $$param; \
				echo "    BEGIN RUN"; \
				echo "    GRAPHS   : $(NUM_GRAPHS)           "; \
				echo "    SIZE     : $$1          "; \
				echo "    MODEL IS : $${model}    "; \
				echo "    THREADS  : $$thread     "; \
				OMP_NUM_THREADS=$$thread $$MODEL $(NUM_GRAPHS) $$1  ; \
				echo "    RUN ENDED"; \
			done; \
		python3 ./utils/src/model_checker.py -o "results/omp-$${model}/distances" -g "tests/groundtruths"; \
		done; \
	done;

# Runs the baseline model and generate the groundtruths. 
run_base:
	@for param in $(SIZE); do \
		echo "    BEGIN RUN"; \
		echo "    GRAPHS   : $(NUM_GRAPHS) "; \
		echo "    SIZE     : $$param      "; \
		echo "    MODEL    : baseline     "; \
		set  -- $$param; \
		./base.o $(NUM_GRAPHS) $$param; \
		echo "RUN ENDED"; \
	done; \
	
# Runs all the CUDA models using different thread numbers
# and different sizes of the graphs. Only the 500 and 1000
# nodes are ran here. 
run_cuda:
	@for model in $(MODELS_CUDA); do \
		MODEL="./cuda_$${model}.o"; \
		for size in $(SIZE); do \
				echo "    BEGIN RUN"; \
				echo "    GRAPHS   : $(NUM_GRAPHS)"; \
				echo "    SIZE     : $${size}     "; \
				echo "    MODEL IS : $${model}    "; \
				echo "    THREADS  : 512     "; \
				$$MODEL $(NUM_GRAPHS) $${size} 512; \
				echo "    RUN ENDED"; \
		done; \
		python3 ./utils/src/model_checker.py -o "results/cuda-$${model}/distances" -g "tests/groundtruths"; \
	done; 
################################################################################################################


#											TEST GEN TARGETS
################################################################################################################
# Generates a certain number of graphs 25 here of all the sizes specified in PARAMETERS
# The parameters can be changed looking at ./utils/src/generate_graphs.py
gen_pos_graphs:
	@for param in $(SIZE); do \
		python3 ./utils/src/generate_graphs.py -n $(NUM_GRAPHS) -s $${param} --negative_edges 0.0; \
	done; \

gen_neg_graphs:
	@for param in $(SIZE); do \
		python3 ./utils/src/generate_graphs.py -n $(NUM_GRAPHS) -s $${param} --negative_edges 1.0; \
	done; \

gen_big_graph:
	python3 ./utils/src/generate_graphs.py -n 1 -s 3000 --negative_edges 1.0;
################################################################################################################


# 											RESULTS CHECKHER TARGETS
################################################################################################################
# For all cuda and omp model specified check the distances saved. If you run after run_cuda or
# run_omp it will check the result with the higher number of threads.
check:
	@for model in $(MODELS_OMP); do \
		echo "CHECKING RESULTS FOR MODEL: omp_$${model}\n"; \
		python3 ./utils/src/model_checker.py -o "results/omp-$${model}/distances" -g "tests/groundtruths"; \
		echo "\n"; \
	done; \

	@for model in $(MODELS_CUDA); do \
		echo "CHECKING RESULTS FOR MODEL: cuda_$${model}\n"; \
		python3 ./utils/src/model_checker.py -o "results/cuda-$${model}/distances" -g "tests/groundtruths"; \
		echo "\n"; \
	done; 
################################################################################################################


# 											EXPERIMENTS TARGETS
################################################################################################################
# EXPERIMENT 1.
run_exp1: gen_pos_graphs all run_base run_omp run_cuda
	echo "EXPERIMENT 1 terminated\n"; \
	mkdir ./res_exp1; \
	mv ./results/* ./res_exp1; \
	rm -r ./tests; \
	rm -r ./results; \
	echo "-----------------------------------------------------------------------------------------------------------"; 

# EXPERIMENT 2.
run_exp2: gen_neg_graphs all run_base run_omp run_cuda
	echo "EXPERIMENT 2 terminated\n"; \
	mkdir ./res_exp2; \
	mv ./results/* ./res_exp2; \
	rm -r ./tests; \
	rm -r ./results; \
	echo "------------------------------------------------------------------------------------------------------";

# EXPERIMENT 3.
exp3: gen_big_graph all
	echo "\nRUN BASELINE\n"; \
	./base.o 1 3000; \
	echo "RUN ENDED\n"; \
	echo "\nRUN CUDA\n"; \
	./cuda_simple.o 1 3000 512; \
	echo "---------------------------------------------------------------------------------------"; \
	./cuda_frontier.o 1 3000 512; \
	echo "RUN ENDED\n"; \
	echo "\nRUN OMP\n"; \
	OMP_NUM_THREADS=1 ./bf_frontier.o 1 3000; \
	echo "---------------------------------------------------------------------------------------"; \
	OMP_NUM_THREADS=1 ./bf_simple.o 1 3000; \
	echo "EXPERIMENT 3 terminated"; \

run_exp3: exp3 check
	mkdir ./res_exp3; \
	mv ./results/* ./res_exp3; \
	rm -r ./results; \
	rm -r ./tests; \
	echo "--------------------------------------------------------------------------------------------------";
################################################################################################################

