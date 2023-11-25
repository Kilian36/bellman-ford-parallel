
# This is a comment line
CC=gcc
# CFLAGS will be the options passed to the compiler.
CFLAGS= -c 

all: prog

utils.o: utils.c
	$(CC) $(CFLAGS) utils.c -o utils.o

main.o: main.c
	$(CC) $(CFLAGS) main.c

baseline.o: baseline.c
	$(CC) $(CFLAGS) baseline.c

prog: main.o utils.o baseline.o
	$(CC) utils.o baseline.o main.o -o prog

clean:
	rm -rf *.o 5 