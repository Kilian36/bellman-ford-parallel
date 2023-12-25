#include "functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Function declarations
void bellman_ford_simple(
                        Graph *graph, 
                        int src, 
                        int p, 
                        int *dist,
                        int *negative,
                        double *time
                        );


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
    char *zeros = malloc(100 * sizeof(char));
    char *output_file = malloc(100 * sizeof(char));
    
    char *threads = getenv("OMP_NUM_THREADS");

    double time, avg_time = 0;

    // Create the output folder
    char *command = malloc(100 * sizeof(char));
    snprintf(command, 100, "mkdir -p ./results/omp-simple/distances");
    system(command);

    int src = 0; 
    

    printf("Output folder is: ./results/omp-simple\n");
    printf("Graph source folder is: ./tests/graphs\n");
    printf("Starting the loop\n");
    for (int i = 0; i < atoi(n_graphs); i++)
    {
        get_str(i, atoi(n_graphs), zeros);

        snprintf(graph_file, 100, "./tests/graphs/%s_%s%d.txt", size, zeros, i);
        snprintf(output_file, 100, "./results/omp-simple/distances/%s_%s%d.txt", size, zeros, i);

        Graph graph = read_graph(graph_file);
        int dist[graph.V];
        int negative = 0;

        if (VERBOSE) {
            printf("Output file: %s\n", output_file);
            printf("Graph file: %s\n", graph_file);
        }

        bellman_ford_simple(
                              &graph, 
                              src, 
                              atoi(threads), 
                              dist,
                              &negative,
                              &time
                            );
        
        avg_time += time;

        // Progress bar
        if (!VERBOSE) printf("\rIn progress %f%%", (float)(i+1)/atoi(n_graphs)*100);
        fflush(stdout);

        if (VERBOSE) printf("Elapsed time: %.5f\n", time);

        // Save the distances or (-1 -1) if negative in the output file
        if (negative) save_negative(output_file);
        else save_dist_array(dist, graph.V, output_file);

        // Append the time to the file times.txt
        FILE *fp;
        fp = fopen("./results/omp-simple/times.txt", "a");
        fprintf(fp, "%s %.5f %d\n", size, time, atoi(threads));
        fclose(fp);

        free_graph(&graph);  

    }
    printf("\nAvg time: %f\n",avg_time / (float)atoi(n_graphs));

    // Free memory
    free(graph_file);
    free(output_file);

    printf("\nDone\n");
    printf("\n");
    
    return 0;
}

/*  
    Compute the bellman ford algorithm, using a parallel loop over the inner loop of the edges. 
    The correctness is sensured by the fact that the computation in the last inner loop are completely 
    independent. All the edges are relaxed every time

    @param graph: a struct containing the adjacency matrix (2D array) and the number of nodes.
    @param src  : the starting node of the bellman frod algorithm.
    @param p    : the number of processors.
    
    @param *dist: array to save the distances from the source node.
    @param *negative: variable to check whether the graph has negative cycles.
    @param *time : variable to save the elapsed time.
*/
void bellman_ford_simple(
                        Graph *graph, 
                        int src, 
                        int p,
                        int *dist,
                        int *negative,
                        double *time
                        ) {

    double init_time = omp_get_wtime();
    
    int n = graph->V;


    // Find local task range
    int local_start[p], local_end[p];

    int ave = n / p;
    #pragma omp parallel for
    for (int i = 0; i < p; i++) {
        local_start[i] = ave * i;
        local_end[i] = ave * (i + 1);
        if (i == p - 1) {
            local_end[i] = n;
        }
    }
    
    // Initialize distances
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        dist[i] = INF;
    }
    //root vertex always has distance 0
    dist[src] = 0;

    int iter_num = 0;
    int has_change;
    int local_has_change[p];
    
    #pragma omp parallel
    {
        int my_rank = omp_get_thread_num();

        for (int iter = 0; iter < n; iter++) {

            local_has_change[my_rank] = 0;
            for (int u = 0; u < n; u++) {

                for (int v = local_start[my_rank]; v < local_end[my_rank]; v++) {

                    int weight = graph->adj[u][v];

                    if (dist[u] + weight < dist[v]) {
                            local_has_change[my_rank] = 1;
                            dist[v] = dist[u] + weight;
                    }
                }
                
            }
            #pragma omp barrier
            #pragma omp single
            {
                iter_num++;
                has_change = 0;
                for (int rank = 0; rank < p; rank++) {
                        if (local_has_change[rank] && (iter == n-1)) {
                            *negative = 1;
                            break;
                        }
                        if(local_has_change[rank]){
                            has_change = 1;
                        }  
                    }     
            }

            #pragma omp barrier
            if (!has_change) {
                break;
            }
        }
    }

    double end_time = omp_get_wtime();
    *time = end_time - init_time;
}
