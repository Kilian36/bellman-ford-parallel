#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void bellman_ford_omp(Graph *graph, int source, char *filename, double *time); 

void bellman_ford(Graph *graph, int source, int p, char *filename, double *time);

void bellman_ford_locks(Graph *graph, int source, int p, char *filename, double *time);

void bellman_ford_frontier(Graph *graph, int source, int p, char *filename, double *time);