#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INF 1000000
#define VERBOSE 0

typedef struct Graph {
    int V;
    int **adj;
} Graph;

typedef struct Graph1D {
    int V;
    int *adj;
} Graph1D;

/*Utilities for graphs based on adjacency matrix*/
void print_graph(Graph graph);
Graph read_graph(char* filename);
Graph1D read_graph1D(char *filename);

/*Utilities for the array with the results*/
void print_dist_array(int *dist, int n);
void save_dist_array(int *dist, int n, char *filename);
void save_negative(char *filename);
void free_graph(Graph *graph);

/*Function to read correctly the file idx*/
void get_str(int idx, int max_idx, char *zeros);