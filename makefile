# This is a comment line
CC=gcc
# CFLAGS will be the options passed to the compiler.
CFLAGS= -c -Wall -fopenmp

all: prog

utils.o: utils.c
	$(CC) $(CFLAGS) utils.c -o utils.o

main.o: main.c
	$(CC) $(CFLAGS) main.c -o main.o

bf-omp.o: bf-omp.c
	$(CC) $(CFLAGS) bf-omp.c -o bf-omp.o

prog: main.o utils.o bf-omp.o
	$(CC) -fopenmp utils.o bf-omp.o main.o -o prog

clean:
	rm -rf *.o prog

# Define the parameters
PARAMETERS := \
	 5 \
	 50 \

# Define the thread settings
THREAD_SETTINGS := \
	8 \
	4 \
	2 \
	1

run_omp:
	@for param in $(PARAMETERS); do \
		for thread in $(THREAD_SETTINGS); do \
			set -- $$param; \
			echo "Running with parameters: 50 $$1 "; \
			OMP_NUM_THREADS=$$thread ./prog 50 $$1; \
			echo "RUN ENDED"; \
			echo "-----------------------------------------------------------"; \
		done; \
	done \

run_python: run_omp
	python3 ./utils/src/model_checker.py -o "results/distances" -g "tests/groundtruths"


