#include "functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Function declarations
void bellman_ford_locks(
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
and saves the distaninit_timeces in ./results/distances and the times in the
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
        
    char *threads = getenv("OMP_NUM_THREADS");


    // Create the output folder
    char *command = malloc(100 * sizeof(char));
    snprintf(command, 100, "mkdir -p ./results/omp-locks/distances");
    system(command);

    double time, avg_time=0.0;
    int src = 0; 
    
    printf("Output folder is: ./results/omp-locks\n");
    printf("Graph source folder is: ./tests/graphs\n");
    printf("Starting the loop\n");
    for (int i = 0; i < atoi(n_graphs); i++)
    {

        get_str(i, atoi(n_graphs), zeros);

        snprintf(graph_file, 100, "./tests/graphs/%s_%s%d.txt", size, zeros, i);
        snprintf(output_file, 100, "./results/omp-locks/distances/%s_%s%d.txt", size, zeros, i);

        Graph graph = read_graph(graph_file);
        int dist[graph.V];
        int negative = 0;

        if (VERBOSE) {
            printf("Output file: %s\n", output_file);
            printf("Graph file: %s\n", graph_file);
        }
        
        bellman_ford_locks(
                              &graph, 
                              src, 
                              atoi(threads), 
                              dist,
                              &negative,
                              &time
                            );

        avg_time += time;

        // Progress bar
        if (!VERBOSE) { printf("\rIn progress %f%%", (float)(i+1)/atoi(n_graphs)*100);
        fflush(stdout);}

        if (VERBOSE) printf("Elapsed time: %.5f\n", time);

        // Save the distances or (-1 -1) if negative in the output file
        if (negative) save_negative(output_file);
        else save_dist_array(dist, graph.V, output_file);

        // Append the time to the file times.txt
        FILE *fp;
        fp = fopen("./results/omp-locks/times.txt", "a");
        fprintf(fp, "%s %.5f %d\n", size, time, atoi(threads));
        fclose(fp);

        free_graph(&graph);  
    }
    
    printf("\n");
    printf("Average time: %f", avg_time/atoi(n_graphs));

    // Free memory
    free(graph_file);
    free(output_file);

    printf("\nDone\n");
    printf("\n");

    return 0;
}

/*  
    Compute the bellman ford algorithm, using a parallel loop over the edges. 
    The correctness is sensured by the fact that every time a distance is changed a 
    lock is set, so that all the others thread cannot access it until it has finished.

    @param graph: a struct containing the adjacency matrix (2D array) and the number of nodes.
    @param src  : the starting node of the bellman frod algorithm.
    @param p    : the number of processors.
    
    @param *dist: array to save the distances from the source node.
    @param *negative: variable to check whether the graph has negative cycles.
    @param *time : variable to save the elapsed time.
*/
void bellman_ford_locks(
                        Graph *graph, 
                        int src, 
                        int p, 
                        int *dist,
                        int *negative, 
                        double *time
                        ) {

    
    double init_time = omp_get_wtime();
    
    int V = graph->V; // Vertices
    
    // Initialize vectors
    int local_has_change[p], has_change=0; 
    int iter = 0;
    

    omp_lock_t *locks = (omp_lock_t *) malloc(V * sizeof(omp_lock_t));
    #pragma omp parallel for
    for (int i = 0; i < V; i++) {
        omp_init_lock(&locks[i]);
    }

    // Step 1: Initialize distances from src to all other
    // vertices as INFINITE
    #pragma omp parallel for
    for (int i = 0; i < V; i++) {
        dist[i] = INF;
    }
    dist[src] = 0;

    // Step 2: Relax all edges |V| - 1 times. A simple
    // shortest path from src to any other vertex can have
    // at-most |V| - 1 edges 
    #pragma omp parallel 
    {
        for (int i = 1; i <= V - 1; i++) {
            #pragma omp for schedule(static)
            for (int j = 0; j < V; j++) {
                for(int k = 0; k < V; k++) {
                    int weight = graph->adj[j][k];
                    if (dist[j] + weight < dist[k]) {

                        omp_set_lock(&locks[k]);
                        dist[k] = dist[j] + weight;
                        omp_unset_lock(&locks[k]);
                        
                        local_has_change[omp_get_thread_num()] = 1;
                    }
                }
            }

            #pragma omp single 
            {
                has_change = 0;
                iter++;
            
                for (int rank = 0; rank < p; rank++) {
                    has_change |= local_has_change[rank];
                }
            }

            #pragma omp barrier
            if (!has_change) {
                break;
            }

            local_has_change[omp_get_thread_num()] = 0;
        }
    }    

    // Step 3: check for negative-weight cycles.  The above
    // step guarantees shortest distances if graph doesn't
    // contain negative weight cycle.  If we get a shorter
    // path, then there is a cycle.
    if (iter == V-1) {
        has_change = 0;
        #pragma omp parallel
        for (int i = 0; i < V; i++) {
                #pragma omp for schedule(auto) reduction(|:has_change)
            for(int j = 0; j<V; j++) {
                int weight = graph->adj[i][j];
                if (dist[i] + weight < dist[j]) {
                    has_change |= 1;
                }
            }
        }
        *negative = has_change;
    }

    
    
    // Step 4: destroy locks
    #pragma omp parallel for
    for (int i = 0; i < V; i++) {
        omp_destroy_lock(&locks[i]);
    }

    double end_time = omp_get_wtime();

    *time = end_time - init_time;

}