#include "functions.h"
#include <stdio.h>
#include <stdlib.h>

void bellman_ford(struct Graph *graph, int src, int *dist, int *negative);

/*
The two arguments represent the number of graphs and the size of the graphs.
The main function reads the graphs from the folder ./tests/graphs
and saves the distances in ./results/distances and the times in the
file times .
*/
int main(int argc, char **argv) {   
    // read graph from file
    printf("Reading graphs from file\n");

    char *size = argv[2];
    char *n_graphs = argv[1];

    char *graph_file = malloc(100 * sizeof(char));
    char *output_file = malloc(100 * sizeof(char));
    char *zeros = malloc(100 * sizeof(char));

    double time;
    int src = 0; 
    
    printf("Output folder is: .tests/groundtruths\n");
    printf("Graph source folder is: ./tests/graphs\n");
    printf("Starting the loop\n");
    for (int i = 0; i < atoi(n_graphs); i++)
    {   
        get_str(i, atoi(n_graphs), zeros);

        snprintf(graph_file, 100, "./tests/graphs/%s_%s%d.txt", size, zeros, i);
        snprintf(output_file, 100, "./tests/groundtruths/%s_%s%d.txt", size, zeros, i);

        // Create the output folder
        char *command = malloc(100 * sizeof(char));
        snprintf(command, 100, "mkdir -p ./tests/groundtruths");
        system(command);

        
        Graph graph = read_graph(graph_file);
        int dist[graph.V];
        int negative = 0;


        if (!VERBOSE) printf("\rIn progress %4f%%", (float)i/atoi(n_graphs)*100);
        fflush(stdout);
        bellman_ford(
                    &graph, 
                    src, 
                    dist,
                    &negative
        );

        // Save the distances or (-1 -1) if negative in the output file
        if (negative) save_negative(output_file);
        else save_dist_array(dist, graph.V, output_file);

        free_graph(&graph);  
    }

    // Free memory
    free(graph_file);
    free(output_file);
    free(zeros);

    printf("\nDone\n");
    
    return 0;
}

/*
The function takes as input a graph, the source node, an array of distances
and a flag to check if there is a negative cycle. The function computes the
shortest path from the source node to all the other nodes in the graph.
@params:
    struct Graph *graph: the graph
    int src: the source node
    int *dist: the array of distances
    int *negative: the flag to check if there is a negative cycle
*/
void bellman_ford(struct Graph *graph, int src, int *dist, int *negative)
{
    int V = graph->V;
 
    // Step 1: Initialize distances from src to all other
    // vertices as INFINITE
    for (int i = 0; i < V; i++)
        dist[i] = INF;
    dist[src] = 0;
 
    // Step 2: Relax all edges |V| - 1 times. A simple
    // shortest path from src to any other vertex can have
    // at-most |V| - 1 edges
    for (int i = 1; i <= V - 1; i++) {
        for (int j = 0; j < V; j++) {
            for (int k = 0; k < V; k++) {
                int weight = graph->adj[j][k];
                if (dist[j] + weight < dist[k])
                    dist[k] = dist[j] + weight;
            }
        }
    }
 
    // Step 3: check for negative-weight cycles.  The above
    // step guarantees shortest distances if graph doesn't
    // contain negative weight cycle.  If we get a shorter
    // path, then there is a cycle.
    for (int i = 0; i < V; i++) {
        for (int j=0; j < V; j++) {
            int weight = graph->adj[i][j];
            if (dist[i] + weight < dist[j]) {
                *negative = 1;
            }
        }
    }
}