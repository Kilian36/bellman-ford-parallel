#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
    #include "utils.h"
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
    
    // Filenames variables
    char *gfile = (char *) malloc(100 * sizeof(char));
    char *ofile = (char *) malloc(100 * sizeof(char));

    // Structurals values
    int num_threads = 1; //int tile_size = 1; 

    // Counters
    clock_t start, end;

    for(int i = 0; i < atoi(n_graphs); i++) {

        // Read files
        snprintf(gfile, 100, "./tests/graphs/%s_%d.txt", size, i);
        snprintf(ofile, 100, "./results/distances/%s_%d.txt", size, i);

        //Output 
        printf("Output file: %s\n", ofile);

        // Read graphs
        Graph1D graph = read_graph1D(gfile);

        // Start the timer
        start = clock();

        // Create results variables
        const int n = graph.V; 
        int *adj=graph.adj; int *d_adj;
        int dist[n]; int *d_dist;
        int *negative = (int *) malloc(sizeof(int));
        int *d_negative;

        // Initialize 
        *negative = 0;
        dist[0] = 0;
        for(int i=1; i<n; i++) {
            dist[i] = INF;
        } 
        
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

        // Copy back the results                        
        cudaMemcpy(dist, d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(negative, d_negative, sizeof(int), cudaMemcpyDeviceToHost);
        
        
        // Save results
        end = clock();

        double elapsed_time = double(end - start) / CLOCKS_PER_SEC;

        if (*negative) save_negative(ofile);
        else save_dist_array(dist, n, ofile);

        FILE *fp;
        fp = fopen("./results/times.txt", "a");
        fprintf(fp, "%d %.5f %d\n", n, elapsed_time, num_threads);
        fclose(fp);


        free(graph.adj);
        cudaFree(d_dist); cudaFree(d_negative); cudaFree(d_adj);
        // Check for errors
        cudaError_t error;
        error = cudaGetLastError();
        const char *error_str = cudaGetErrorString(error);
        //printf("%s\n", error_str);
    }

    free(gfile); free(ofile);
}