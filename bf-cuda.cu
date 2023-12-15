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
    Bellman-Ford algorithm, parallelized with CUDA, through a monolitic kernel. 

    @param adj: adjacency matrix
    @param dist: distance array
    @param n: number of vertices
    @param negative: flag to indicate negative cycle    
*/
__global__ void bellman_ford(int *adj, int *dist, int n, int *negative) {
    int v = threadIdx.x;
    
    // First k-1 iterations
    for(int k=0; k<n-1; k++) {
        for(int u=0; u<n; u++) {
            if (v < n) {
                if(dist[u] + adj[u * n + v] < dist[v]) {
                    dist[v] = dist[u] + adj[u * n + v];
                }
            }
        }
        __syncthreads();
    }
    
    // Check for negative cycles
    if (v < n) {
        for(int u=0; u<n; u++) {
            if (u < n) {
                if(dist[u] + adj[u * n + v] < dist[v]) {
                    *negative = 1;
                }
            }
        }
    }
}

/*
    Bellman-Ford algorithm, parallelized with CUDA, through a tiled kernel. 

    @param adj: adjacency matrix
    @param dist: distance array
    @param n: number of vertices
    @param negative: flag to indicate negative cycle
    @param tile_size: size of the tile, it must be V / N_THREADS + 1.
*/
__global__ void bellman_ford_tiled(int *adj, int *dist, int n, int *negative, int tile_size) {
    int v = threadIdx.x; 
    int loc_idx = v * tile_size;

    // First k-1 iterations
    for(int k=0; k<n-1; k++) {
        for(int u=0; u<n; u++) { // loop over the starting nodes
            for(int t=0; t<tile_size; t++) { // Loop over the arrival nodes in the tile
                
                if(loc_idx + t < n) {
                    if(dist[u] + adj[u * n + loc_idx + t] < dist[loc_idx + t]) {
                        dist[loc_idx + t] = dist[u] + adj[u * n + loc_idx + t];
                    }
                }

            }
        }
        __syncthreads();
    }

    // Check for negative cycles
    for(int u=0; u<n; u++) {
        for(int t=0; t<tile_size; t++) {
            if(v + t < n) {
                if(dist[u] + adj[u * n + loc_idx + t] < dist[loc_idx + t]) {
                    *negative = 1;
                }
            }
        }
    }
    
}

int main(int argc, char **argv) {
    char *size = argv[2];
    char *n_graphs = argv[1];

    char *gfile = (char *) malloc(100 * sizeof(char));
    char *ofile = (char *) malloc(100 * sizeof(char));

    int num_threads = 100;

    int tile_size = 1; 

    clock_t start, end;

    printf("Starting the loop\n");
    printf("\n");

    for (int i = 0; i < atoi(n_graphs); i++) {
        snprintf(gfile, 100, "./tests/graphs/%s_%d.txt", size, i);
        snprintf(ofile, 100, "./results/distances/%s_%d.txt", size, i);

        printf("Output file: %s\n", ofile);
        
        // Read graph
        Graph1D graph = read_graph1D(gfile);

        // Start the timer
        start = clock();

        int n = graph.V; // Vertices

        if ((TILED) && (n > num_threads)) {tile_size = n / num_threads + 1;} 

        int *dist     = (int *) malloc(n * sizeof(int));
        int *negative = (int *) malloc(sizeof(int));
        *negative = 0;
        
        // Initialize the distances array
        for (int i = 0; i < n; i++) {
            dist[i] = INF;
        }
        
        // Source vertex
        int src = 0;
        dist[src] = 0;

        
        // Move pointers to cuda device
        int *d_adj, *d_dist, *d_negative;
        
        cudaMalloc((void **)&d_adj, n * n * sizeof(int));
        cudaMalloc((void **)&d_dist, n * sizeof(int));
        cudaMalloc((void **)&d_negative, sizeof(int));

        cudaMemcpy(d_adj, graph.adj, n * n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dist, dist, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_negative, negative, sizeof(int), cudaMemcpyHostToDevice);
        
        if (TILED) bellman_ford_tiled<<<1, num_threads>>>
                                    (d_adj, 
                                     d_dist, 
                                     n, 
                                     d_negative, 
                                     tile_size
                                    );

        else bellman_ford<<<1, num_threads>>>
                                    (d_adj, 
                                     d_dist,
                                     n, 
                                     d_negative
                                    );
                                    
        // Copy back the results                        
        cudaMemcpy(dist, d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(negative, d_negative, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Save results
        end = clock();

        double elapsed_time = double(end - start) / CLOCKS_PER_SEC;

        if (*negative) {
            save_negative(ofile);
        }
        else {
            save_dist_array(dist, n, ofile);
        }

        FILE *fp;
        fp = fopen("./results/times.txt", "a");
        fprintf(fp, "%s %.5f %d\n", size, elapsed_time, num_threads);
        fclose(fp);
        
        // Free memory
        free(dist); free(negative); free(graph.adj);
        cudaFree(d_dist); cudaFree(d_negative); cudaFree(d_adj);
        
        // Check for errors
        cudaError_t error;
        error = cudaGetLastError();
        const char *error_str = cudaGetErrorString(error);
        printf("%s\n", error_str);
    
    }

    free(gfile);
    free(ofile);

    printf("Done\n");


    return 0;
}
