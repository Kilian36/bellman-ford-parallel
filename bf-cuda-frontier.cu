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

    @param *adj  : the adjacency matrix of the graph.
    @param src  : the starting node of the bellman frod algorithm.
    @param p    : the number of processors.
    @param n    : the number of nodes of the graph.
    
    @param *dist: array to save the distances from the source node.
    @param *negative: variable to check whether the graph has negative cycles.
    @param *time : variable to save the elapsed time.
*/
__global__ void bellman_ford_frontier(
                                int *adj, 
                                int *dist, 
                                int *negative, 
                                int n, 
                                int tile_size, 
                                int src
                ) {
    
    __shared__ int frontier_size; 
    int front_idx = 0; int idx = 0; 

    extern __shared__ int array[]; // Get shared dynamic memory
    
    int *frontier      = (int*) array; 
    int *next_frontier = (int*) &array[n];
    int *has_changed   = (int*) &array[2*n];

    int local_start, local_end;
    
    int tid = threadIdx.x; 
    
    if (tid * tile_size < n) {
        local_start = tile_size * tid;
        local_end = tile_size * (tid + 1);

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
}

void run_dummy_graph(int *d_adj, int *d_dist, int *d_negative, int n, int tile_size, int num_threads) {
    
    int *dist     = (int *) malloc(n * sizeof(int));
    int *negative = (int *) malloc(sizeof(int));
    int *adj      = (int *) malloc(n * n * sizeof(int));
    
    // Init variables
    *negative = 0;

    // Initialize distances and copy vector to GPU
    for(int j=0; j<n; j++) {
        dist[j] = INF;
    }
    dist[0] = 0;

    for (int i=0; i<n*2; i++) {
        adj[i] = INF;  
    }


    // Copy data to the GPU
    cudaMemcpy(d_negative, negative, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj, adj, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist, n * sizeof(int), cudaMemcpyHostToDevice);

    bellman_ford_frontier<<<1, num_threads, 3 * n *sizeof(int)>>>
                            (d_adj,  
                            d_dist, 
                            d_negative, 
                            n,
                            tile_size,
                            0
                            );

}

int main(int argc, char **argv) {
    
    // Get input args
    char *size = argv[2];
    char *n_graphs = argv[1];
    char *threads = argv[3];
    
    // Filenames variables
    char *gfile = (char *) malloc(100 * sizeof(char));
    char *ofile = (char *) malloc(100 * sizeof(char));
    char *zeros = (char *) malloc(100 * sizeof(char));

    // Convert types
    int num_threads = atoi(threads); 
    const int n = atoi(size);
    int src = 0;
    int tile_size = 1;

    // Counters
    clock_t start, end;
    double avg_time = 0;

    if (num_threads > n) num_threads=n;
    if (n > num_threads) tile_size = n / num_threads + 1; 
    
    //  PreAllocate CUDA memory
    int *d_adj, *d_dist, *d_negative;
    cudaMalloc((void **)&d_dist, n * sizeof(int));
    cudaMalloc((void **)&d_negative, sizeof(int));
    cudaMalloc((void **)&d_adj, n * n * sizeof(int));
    
    //run_dummy_graph(d_adj, d_dist, d_negative, n, tile_size, num_threads);

    for(int i = 0; i < atoi(n_graphs); i++) {
        
        get_str(i, atoi(n_graphs), zeros);
        
        snprintf(gfile, 100, "./tests/graphs/%s_%s%d.txt", size, zeros, i);
        snprintf(ofile, 100, "./results/cuda-frontier/distances/%s_%s%d.txt", size, zeros, i);
        
        // Create the output folder name
        char *command = (char *) malloc(100 * sizeof(char));
        snprintf(command, 100, "mkdir -p ./results/cuda-frontier/distances");
        system(command);

        //Output 
        if(VERBOSE) {
            printf("Output file: %s\n", ofile);
            printf("Graph to read file: %s\n", gfile);
        }

        // Read graph
        Graph1D graph = read_graph1D(gfile);

        // Start the timer
        start = clock();

        int *dist     = (int *) malloc(n * sizeof(int));
        int *negative = (int *) malloc(sizeof(int));
        
        // Init variables
        *negative = 0;

        // Initialize distances and copy vector to GPU
        for(int j=0; j<n; j++) {
            dist[j] = INF;
        }
        dist[src] = 0; 
        
        // Copy data to the GPU
        cudaMemcpy(d_negative, negative, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_adj, graph.adj, n * n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dist, dist, n * sizeof(int), cudaMemcpyHostToDevice);

        bellman_ford_frontier<<<1, num_threads, 3 * n *sizeof(int)>>>
                            (d_adj,  
                            d_dist, 
                            d_negative, 
                            n,
                            tile_size,
                            src
                            );
        
        // Copy back the results                        
        cudaMemcpy(dist, d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(negative, d_negative, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Save results
        end = clock();

        double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
        avg_time += elapsed_time;

        // Progress bar
        if (!VERBOSE) { 
            printf("\rIn progress %f%%", (float)(i+1)/atoi(n_graphs)*100);
            fflush(stdout);
        }

        if(VERBOSE) printf("Elapsed time: %f ", elapsed_time);

        if (*negative) save_negative(ofile);
        else save_dist_array(dist, n, ofile);

        FILE *fp;
        fp = fopen("./results/cuda-frontier/times.txt", "a");
        fprintf(fp, "%d %.5f %d\n", n, elapsed_time, num_threads);
        fclose(fp);

        free(graph.adj);

        // Check for errors
        cudaError_t error;
        error = cudaGetLastError();
        const char *error_str = cudaGetErrorString(error);
    }
    
    cudaFree(d_dist); cudaFree(d_negative); cudaFree(d_adj);

    printf("\n");
    printf("Average time: %f\n", avg_time / atoi(n_graphs));

    free(gfile); free(ofile); free(zeros);
    return 0;
}