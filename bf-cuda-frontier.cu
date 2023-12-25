#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
    #include "functions.h"
}


#define BLOCK_SIZE 32
#define TILED 1

/*
Copy front2 values in front1 to begin the new iteration of the loop 
and reinitialize front2 with -1's.
*/
__device__ void swap_and_restart(int *front1, int *front2, int *new_size, int size) {
    int real_idx = 0;
    for (int i = 0; i < size; i++ )
    {
        if (front2[i] != -1) {
            front1[real_idx] = front2[i];
            real_idx++;
        }
        front2[i] = -1;
    }
    *new_size = real_idx;  
}


/*  
    Compute the bellman ford algorithm efficiently, using the forntier method. This helps to reduce the 
    number of iterations of the algorithm, since it only relax over the nodes that have been modified. 
    The adventage of the cuda version is that it can exploit fast shared memory among the same block and an 
    extremely high number of threads.

    @param graph: a struct containing the adjacency matrix (2D array) and the number of nodes.
    @param src  : the starting node of the bellman frod algorithm.
    @param p    : the number of processors.
    
    @param *dist: array to save the distances from the source node.
    @param *negative: variable to check whether the graph has negative cycles.
    @param *time : variable to save the elapsed time.
*/
__global__ void bellman_ford_frontier(
                                int *adj, 
                                int *dist, 
                                int *negative, 
                                int n, 
                                int p, 
                                int src
                ) {
    
    __shared__ int frontier_size; 
    int front_idx = 0; int idx = 0; 

    extern __shared__ int array[]; // Get shared dynamic memory
    
    int *frontier      = (int*) array; 
    int *next_frontier = (int*) &array[n];
    int *has_changed   = (int*) &array[2*n];

    int local_start, local_end, avg = n / p;
    
    int tid = threadIdx.x; 
    
    local_start = avg * tid;
    
    if (tid != p-1) local_end = avg * (tid + 1);
    else local_end = n;

    if (tid==0) frontier_size=1;

    // Initialize the variables
    for(int i=local_start; i<local_end; i++) {
        has_changed[i] = 0;
        frontier[i] = -1;
        next_frontier[i] = -1;
    }
    frontier[0] = src;
    
    while (frontier_size>0 && idx<n) {
        idx++;
        front_idx=local_start;
        
        for(int i=0; i<frontier_size; i++) {
            int u = frontier[i];
            
            for(int v = local_start; v<local_end; v++) {
                int idx1D = n*u + v;

                if(dist[v] > dist[u] + adj[idx1D]){
                    if (has_changed[v]) { // Check whether the node v was already modified 
                        dist[v] = dist[u] + adj[idx1D];
                    }
                    else { // If note update
                        next_frontier[front_idx] = v;
                        has_changed[v] = 1;
                        front_idx++;
                        dist[v] = dist[u] + adj[idx1D];
                    }    
                } 
            }
            __syncthreads();
            
        }

        // Sequantial component of the program
        if (tid == 0) {

            // Check for negative cycles
            if (idx == n) {
                for (int i = 0; i < n; i++) {
                    if (next_frontier[i] != -1) {
                    *negative = 1;
                    }
                }
            } 

            // Swap frontiers and restart
            swap_and_restart(frontier, next_frontier, &frontier_size, n);

            // Set the local changes to zero
            for(int i=0; i<n; i++) has_changed[i]=0;    
        }
        __syncthreads();

    }
}

int main(int argc, char **argv) {
    
    char *size = argv[2];
    char *n_graphs = argv[1];
    char *threads = argv[3];
    
    // Filenames variables
    char *gfile = (char *) malloc(100 * sizeof(char));
    char *ofile = (char *) malloc(100 * sizeof(char));
    char *zeros = (char *) malloc(100 * sizeof(char));

    // Structurals values
    int num_threads = atoi(threads); 
    int scr = 0;

    // Counters
    clock_t start, end;
    double avg_time = 0;

    if (num_threads > atoi(size)) num_threads=atoi(size);

    for(int i = 0; i < atoi(n_graphs); i++) {
        
        get_str(i, atoi(n_graphs), zeros);
        
        snprintf(gfile, 100, "./tests/graphs/%s_%s%d.txt", size, zeros, i);
        snprintf(ofile, 100, "./results/cuda-frontier/distances/%s_%s%d.txt", size, zeros, i);
        
        // Create the output folder
        char *command = (char *) malloc(100 * sizeof(char));
        snprintf(command, 100, "mkdir -p ./results/cuda-frontier/distances");
        system(command);

        //Output 
        if(VERBOSE) {
            printf("Output file: %s\n", ofile);
            printf("Graph to read file: %s\n", gfile);
        }

        // Read graphs
        Graph1D graph = read_graph1D(gfile);

        // Start the timer
        start = clock();

        // Create results variables
        const int n = graph.V; 
        // Make sure there is at most one thread per node

        int *adj=graph.adj; int *d_adj;
        int dist[n]; int *d_dist;
        int *negative = (int *) malloc(sizeof(int));
        int *d_negative;

        // Initialize 
        *negative = 0;
        for(int j=0; j<n; j++) {
            dist[j] = INF;
        }
        dist[scr] = 0; 
        
        cudaMalloc((void **)&d_dist, n * sizeof(int));
        cudaMalloc((void **)&d_negative, sizeof(int));
        cudaMalloc((void **)&d_adj, n * n * sizeof(int));
        
        cudaMemcpy(d_dist, dist, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_negative, negative, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_adj, adj, n * n * sizeof(int), cudaMemcpyHostToDevice);

        bellman_ford_frontier<<<1, num_threads, 3 * n * sizeof(int)>>>
                            (d_adj,  
                            d_dist, 
                            d_negative, 
                            n,
                            num_threads,
                            0
                            );
        
        // Progress bar
        if (!VERBOSE) { 
            printf("\rIn progress %f%%", (float)(i+1)/atoi(n_graphs)*100);
            fflush(stdout);
        }

        // Copy back the results                        
        cudaMemcpy(dist, d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(negative, d_negative, sizeof(int), cudaMemcpyDeviceToHost);
        
        
        // Save results
        end = clock();

        double elapsed_time = double(end - start) / CLOCKS_PER_SEC;

        avg_time += elapsed_time;

        if(VERBOSE) printf("Elapsed time: %f ", elapsed_time);

        if (*negative) save_negative(ofile);
        else save_dist_array(dist, n, ofile);

        FILE *fp;
        fp = fopen("./results/cuda-frontier/times.txt", "a");
        fprintf(fp, "%d %.5f %d\n", n, elapsed_time, num_threads);
        fclose(fp);


        free(graph.adj);
        cudaFree(d_dist); cudaFree(d_negative); cudaFree(d_adj);
        // Check for errors
        cudaError_t error;
        error = cudaGetLastError();
        const char *error_str = cudaGetErrorString(error);
        printf("%s\n", error_str);
    }

    printf("\n");
    printf("Average time: %f\n", avg_time / atoi(n_graphs));

    free(gfile); free(ofile); free(zeros);
    return 0;
}