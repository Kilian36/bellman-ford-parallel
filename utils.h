#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INT_MAX 1000000

struct Edge {
    int src, dest, weight;
};

struct Graph {
    int V, E;
    struct Edge *edge;
};

void print_graph(struct Graph graph);

struct Graph create_graph(int V, int E);

struct Graph read_graph(char* filename);

void print_dist_array(int *dist, int n);

void save_dist_array(int *dist, int n, char *filename);

void save_negative(char * filename);