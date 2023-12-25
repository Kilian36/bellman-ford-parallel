#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
    #include "functions.h"
}

#define BLOCK_SIZE 32

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
    char *threads = argv[3];

    char *gfile = (char *) malloc(100 * sizeof(char));
    char *ofile = (char *) malloc(100 * sizeof(char));
    char *zeros = (char *) malloc(100 * sizeof(char));

    int num_threads = atoi(threads);

    int tile_size = 1; 

    clock_t start, end;
    double avg_time = 0;

    if (num_threads > atoi(size)) num_threads=atoi(size);

    printf("Starting the loop\n");
    printf("\n");
    
    for (int i = 0; i < atoi(n_graphs); i++) {
        get_str(i, atoi(n_graphs), zeros);
        
        snprintf(gfile, 100, "./tests/graphs/%s_%s%d.txt", size, zeros, i);
        snprintf(ofile, 100, "./results/cuda-simple/distances/%s_%s%d.txt", size, zeros, i);

        // Create the output folder
        char *command = (char *) malloc(100 * sizeof(char));
        snprintf(command, 100, "mkdir -p ./results/cuda-simple/distances");
        system(command);

        if (VERBOSE) {
            printf("Output file: %s\n", ofile);
            printf("Graph to read file: %s\n", gfile);
        }

        // Read graph
        Graph1D graph = read_graph1D(gfile);

        
        // Start the timer
        start = clock();

        int n = graph.V; // Vertices
        int src = 0;

        if (n > num_threads) {tile_size = n / num_threads + 1;} 

        int *dist     = (int *) malloc(n * sizeof(int));
        int *negative = (int *) malloc(sizeof(int));
        *negative = 0;
        
        // Initialize the distances array
        for (int j = 0; j < n; j++) {
            dist[j] = INF;
        }
        dist[src] = 0;
        // Source vertex
        dist[src] = 0;

        
        // Move pointers to cuda device
        int *d_adj, *d_dist, *d_negative;
        
        cudaMalloc((void **)&d_adj, n * n * sizeof(int));
        cudaMalloc((void **)&d_dist, n * sizeof(int));
        cudaMalloc((void **)&d_negative, sizeof(int));

        cudaMemcpy(d_adj, graph.adj, n * n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dist, dist, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_negative, negative, sizeof(int), cudaMemcpyHostToDevice);

    
        bellman_ford_tiled<<<1, num_threads>>>
                                    (d_adj, 
                                     d_dist, 
                                     n, 
                                     d_negative, 
                                     tile_size
                                    );
                                    
        // Copy back the results                        
        cudaMemcpy(dist, d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(negative, d_negative, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Save results
        end = clock();

        // Progress bar
        if (!VERBOSE) { printf("\rIn progress %f%%", (float)(i+1)/atoi(n_graphs)*100);
        fflush(stdout);}
       
        double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
        avg_time += elapsed_time;

        if(VERBOSE) printf("Elapsed time: %f ", elapsed_time);
 
        if (*negative) {
            save_negative(ofile);
        }
        else {
            save_dist_array(dist, n, ofile);
        }
    
        FILE *fp;
        fp = fopen("./results/cuda-simple/times.txt", "a");
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

    printf("\n");
    printf("Average time: %f\n", avg_time / atoi(n_graphs));

    free(gfile); free(ofile); free(zeros);


    printf("Done\n");

    
    return 0;

}