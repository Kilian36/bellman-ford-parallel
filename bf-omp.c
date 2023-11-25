#include "bf-omp.h"

void bellman_ford_omp(struct Graph graph, int src, char *filename, double *time){
    double init_time, end_time; 
    int V = graph.V;
    int E = graph.E;
    int dist[V];
 
    init_time = omp_get_wtime();

    // Step 1: Initialize distances from src to all other
    // vertices as INFINITE
    for (int i = 0; i < V; i++) {
        dist[i] = INT_MAX;
    }
    dist[src] = 0;
    
    // Step 2: Relax all edges |V| - 1 times. A simple
    // shortest path from src to any other vertex can have
    // at-most |V| - 1 edges
    for (int i = 1; i <= V - 1; i++) {
        #pragma omp parallel  
        {
        #pragma omp for schedule(static)
        for (int j = 0; j < E; j++) {
            int u = graph.edge[j].src;
            int v = graph.edge[j].dest;
            int weight = graph.edge[j].weight;
            if (dist[u] != INT_MAX
                && dist[u] + weight < dist[v])
                dist[v] = dist[u] + weight;
        }
        }
    }
 
    // Step 3: check for negative-weight cycles.  The above
    // step guarantees shortest distances if graph doesn't
    // contain negative weight cycle.  If we get a shorter
    // path, then there is a cycle.
    int negative = 0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < E; i++) {
        
        int u = graph.edge[i].src;
        int v = graph.edge[i].dest;
        int weight = graph.edge[i].weight;
        if (dist[u] != INT_MAX
            && dist[u] + weight < dist[v]) {
            end_time = omp_get_wtime();
            #pragma omp atomic
                negative++;
        }
    }
    if (negative) {
        save_negative(filename);
        *time = end_time - init_time;
        return;
    }
    else {
        end_time = omp_get_wtime();
        save_dist_array(dist, V, filename);

        *time = end_time - init_time;
    }
    return;
}