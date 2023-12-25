#include "functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Function declarations
void swap_and_init(
                 int *front1, 
                 int *front2, 
                 int *new_size, 
                 int n
                );

void bellman_ford_frontier(
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
    char *output_file = malloc(100 * sizeof(char));
    char *zeros = malloc(100 * sizeof(char));
    char *threads = getenv("OMP_NUM_THREADS");

    double time, avg_time = 0;


    // Create the output folder
    char *command = malloc(100 * sizeof(char));
    snprintf(command, 100, "mkdir -p ./results/omp-frontier/distances");
    system(command);

    int src = 0; 
    
    printf("Output folder is: ./results/omp-frontier\n");
    printf("Graph source folder is: ./tests/graphs\n");
    printf("Starting the loop\n");

    for (int i = 0; i < atoi(n_graphs); i++)
    {
    
        get_str(i, atoi(n_graphs), zeros);

        snprintf(graph_file, 100, "./tests/graphs/%s_%s%d.txt", size, zeros, i);
        snprintf(output_file, 100, "./results/omp-frontier/distances/%s_%s%d.txt", size, zeros, i);

        Graph graph = read_graph(graph_file);
        int dist[graph.V];
        int negative = 0;

        if (VERBOSE) {
            printf("Output file: %s\n", output_file);
            printf("Graph file: %s\n", graph_file);
        }

        bellman_ford_frontier(
                              &graph, 
                              src, 
                              atoi(threads), 
                              dist,
                              &negative,
                              &time
                            );
        
        avg_time += time;

        if (VERBOSE) printf("Elapsed time: %.5f\n", time);

        // Save the distances or (-1 -1) if negative in the output file
        if (negative) save_negative(output_file);
        else save_dist_array(dist, graph.V, output_file);

        // Progress bar
        if (!VERBOSE) printf("\rIn progress %f%%", (float)(i+1)/atoi(n_graphs)*100);
        fflush(stdout);

        // Append the time to the file times.txt
        FILE *fp;
        fp = fopen("./results/omp-frontier/times.txt", "a");
        fprintf(fp, "%s %.5f %d\n", size, time, atoi(threads));
        fclose(fp);

        free_graph(&graph);  
    }

    printf("\n");
    printf("Average time: %.5f\n", avg_time/atoi(n_graphs));

    // Free memory
    free(graph_file);
    free(output_file);

    printf("\nDone\n");
    printf("\n");
    
    return 0;
}

/*
    @param *front1 array to fill with the value in *front2.
    @param *front2 array to fill with -1's.
    @param n the size of both arrays.   
*/
void swap_and_init(int *front1, int *front2, int *new_size, int n) {
    int true_idx = 0;
    for (int i = 0; i < n; i++ )
    {
        if (!(front2[i] == -1)) {
            front1[true_idx] = front2[i];
            true_idx++;
        }
        front2[i] = -1;
    }
    *new_size = true_idx; 
}

/*  
    Compute the bellman ford algorithm efficiently, using the forntier method. This helps to reduce the 
    number of iterations of the algorithm, since it only relax over the nodes that have been modified.

    @param graph: a struct containing the adjacency matrix (2D array) and the number of nodes.
    @param src  : the starting node of the bellman frod algorithm.
    @param p    : the number of processors.
    
    @param *dist: array to save the distances from the source node.
    @param *negative: variable to check whether the graph has negative cycles.
    @param *time : variable to save the elapsed time.
*/
void bellman_ford_frontier
               (Graph *graph, 
                int src, 
                int p, 
                int *dist,
                int *negative, 
                double *time) {
    
    int n = graph->V; // Vertices

    int frontier_size = 1; int next_frontier_size = 0; // Array dims

    int frontier[n], next_frontier[n], has_changed[n]; // Support arrays

    int local_start[p], local_end[p]; // Thread arrays

    // Begin the computation

    double init_time = omp_get_wtime(); // Get time

    // Local task range
    int avg = n / p;
    #pragma omp parallel for
    for (int i = 0; i < p; i++) {

        local_start[i] = avg * i;
        local_end[i] = avg * (i + 1);
        if (i == p - 1) {
            local_end[i] = n;
        }
    }

    // Initialize vectors
    #pragma omp parallel for schedule(static) 
    for (int i = 0; i < n; i++) {
        dist[i] = INF;
        has_changed[i] = 0;
        frontier[i] = -1;
        next_frontier[i] = -1;
    }
    dist[src] = 0;
    frontier[0] = src;
    
    // Bellman-Ford algorithm
    #pragma omp parallel default(shared) private(next_frontier_size)  
    {
        int idx = 0;

        while (frontier_size > 0 && idx++ < n) {  

            int thread_id = omp_get_thread_num();
            next_frontier_size = local_start[thread_id];
            for(int i=0; i<frontier_size; i++) {
                
                int u = frontier[i]; // Select current node in frontier
                {
                    
                    for(int v=local_start[thread_id]; v<local_end[thread_id]; v++){ 
            
                        if(dist[v] > dist[u] + graph->adj[u][v]){
                            if (has_changed[v]) { // Check whether the node v was already modified 
                                dist[v] = dist[u] + graph->adj[u][v];
                            }
                            
                            else { // If note update
                                next_frontier[next_frontier_size] = v;
                                has_changed[v] = next_frontier_size;
                                next_frontier_size++;
                                dist[v] = dist[u] + graph->adj[u][v];
                            }
                            
                        }
                        
                    }
                }
            }
            #pragma omp barrier 

            // Reset has_changed            
            for(int k=local_start[thread_id]; k<local_end[thread_id]; k++) 
                has_changed[k]=0;

            #pragma omp single
            {
                if (idx == n) {
                    for (int i = 0; i < n; i++) {
                        if (next_frontier[i] != -1) {
                        *negative = 1;
                        break;
                        }
                    }
                } 
            
                // swap front1 and front2
                swap_and_init(frontier, next_frontier, &frontier_size, n); 
            }
            #pragma omp barrier
        }
    }
    
    double end_time = omp_get_wtime();

    *time = end_time - init_time;
}



