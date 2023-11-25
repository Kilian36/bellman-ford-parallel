#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void bellman_ford_omp(struct Graph graph, int source, char *filename, double *time); 
