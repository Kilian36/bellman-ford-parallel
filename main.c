#include <stdio.h>
#include <stdlib.h>
#include "bf-omp.h"
#include <time.h>

#define VERBOSE 0
/*
The two arguments represent the number of graphs and the size of the graphs.
The main function reads the graphs from the folder ./tests/graphs
and saves the distances in ./results/distances and the times in the
file times .
*/

int main(int argc, char **argv)
{   
    // read graph from file
    printf("Reading graphs from file\n");

    char *size = argv[2];
    char *n_graphs = argv[1];
    char *model = argv[3];

    char *graph_file = malloc(100 * sizeof(char));
    char *output_file = malloc(100 * sizeof(char));
    char *threads = getenv("OMP_NUM_THREADS");

    double time;
    
    printf("Starting the loop\n");
    printf("\n");
    for (int i = 0; i < atoi(n_graphs); i++)
    {
    
        snprintf(graph_file, 100, "./tests/graphs/%s_%d.txt", size, i);
        snprintf(output_file, 100, "./results/distances/%s_%d.txt", size, i);
        Graph graph = read_graph(graph_file);

        printf("Output file: %s\n", output_file);

        //bellman_ford(&graph, 0, atoi(threads),output_file, &time);
        bellman_ford_frontier(&graph, 0, atoi(threads), output_file, &time);


        // bellman_ford_omp(&graph, 0, output_file, &time);
        if (VERBOSE)
            printf("Elapsed time: %.5f\n", time);

        // Append the time to the file times.txt

        FILE *fp;
        fp = fopen("./results/times.txt", "a");
        fprintf(fp, "%s %.5f %d\n", size, time, atoi(threads));
        fclose(fp);

        free_graph(&graph);
        
    }

    free(graph_file);
    free(output_file);

    printf("Done\n");
    
    return 0;
}
