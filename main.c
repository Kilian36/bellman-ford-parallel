// Let's read and pring a graph from graphs/graph_30_0.txt
#include "baseline.h"
#include <stdio.h>
#include <stdlib.h>



int main(int argc, char** argv)
{
    // read graph from file
    printf("Reading graph from file\n");
    char *size = argv[1];
    int n_graphs = atoi(argv[2]);

    char *graph_file = malloc(100 * sizeof(char));
    char *output_file = malloc(100 * sizeof(char));
    
    printf("Starting the loop\n");
    for(int i=0; i<n_graphs; i++) {
        snprintf(graph_file, 100, "./tests/graphs/%s_%d.txt", size, i);
        snprintf(output_file, 100, "./tests/groundtruths/%s_%d.txt", size, i);
        struct Graph graph = read_graph(graph_file);

        bellman_ford(graph, 0, output_file);

        free(graph.edge);
    }
    
    free(graph_file);
    free(output_file);

    printf("Done\n");
    return 0;
}
