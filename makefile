# Define the parameters
PARAMETERS := \
	 50 \
	 500
	 	
THREAD_SETTINGS := \
	1 \
	2 \
	4 \
	8 
	
# Clean the old results if any
clean_results:
	rm -rf results/times.txt && \
	rm -rf results/distances/*.txt && \
	rm -rf results/*.png \
	rm -rf results/statistics.csv

all:
	gcc -o prog utils.c bf-omp.c main.c -fopenmp;

run: 
	@for param in $(PARAMETERS); do \
		for thread in $(THREAD_SETTINGS); do \
			set -- $$param; \
			echo "Running with 25 graphs of size $$1 "; \
			OMP_NUM_THREADS=$$thread ./prog 25 $$1; \
			echo "RUN ENDED"; \
			echo "-----------------------------------------------------------"; \
		done; \
	done && \
	python3 ./utils/src/model_checker.py -o "results/distances" -g "tests/groundtruths"
	
run_hard: run
	@for thread in $(THREAD_SETTINGS); do  \
		echo "Running with parameters: 5 1000 "; \
		OMP_NUM_THREADS=$$thread ./prog 5 1000; \
		echo "RUN ENDED"; \
		echo "-----------------------------------------------------------"; \
	done 
	python3 ./utils/src/model_checker.py -o "results/distances" -g "tests/groundtruths";
	python3 ./utils/src/plot_results.py;  

run_trivial: 
	OMP_NUM_THREADS=2 ./prog 25 5;
	python3 ./utils/src/model_checker.py -o "results/distances" -g "tests/groundtruths";


run_cuda:
	nvcc -o cuda_prog utils.c bf-cuda.cu; \
	./cuda_prog 25 500; \
	echo "Run with 1000 nodes graphs"; \
	./cuda_prog 5 1000; \
	python3 ./utils/src/model_checker.py -o "results/distances" -g "tests/groundtruths";
	

clean:
	rm -rf *.o prog \

