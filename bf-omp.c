#include "bf-omp.h"

void bellman_ford(Graph *graph, int src, int p, char *filename, double *time) {
    double init_time = omp_get_wtime();
    
    int local_start[p], local_end[p];
    int has_negative_cycle = 0;

    int n = graph->V;

    int dist[n];

    //step 2: find local task range
    int ave = n / p;
    #pragma omp parallel for
    for (int i = 0; i < p; i++) {
        local_start[i] = ave * i;
        local_end[i] = ave * (i + 1);
        if (i == p - 1) {
            local_end[i] = n;
        }
    }
    
    //step 3: bellman-ford algorithm
    //initialize distances
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        dist[i] = INF;
    }
    //root vertex always has distance 0
    dist[0] = 0;

    int iter_num = 0;
    int has_change;
    int local_has_change[p];
    
    #pragma omp parallel
    {
        int my_rank = omp_get_thread_num();
        //bellman-ford algorithm
        for (int iter = 0; iter < n - 1; iter++) {
            local_has_change[my_rank] = 0;
            for (int u = 0; u < n; u++) {
                for (int v = local_start[my_rank]; v < local_end[my_rank]; v++) {
                    int weight = graph->adj[u][v];
                    if (weight < INF) {
                        int new_dis = dist[u] + weight;
                        if (new_dis < dist[v]) {
                            local_has_change[my_rank] = 1;
                            dist[v] = new_dis;
                        }
                    }
                }
            }
            #pragma omp barrier
            #pragma omp single
            {
                iter_num++;
                has_change = 0;
                for (int rank = 0; rank < p; rank++) {
                    has_change |= local_has_change[rank];
                }
            }
                if (!has_change) {
                    break;
                }
        }
    }

    //do one more iteration to check negative cycles
    if (iter_num == n - 1) {
        has_change = 0;
        for (int u = 0; u < n; u++) {
            #pragma omp parallel for reduction(|:has_change)
            for (int v = 0; v < n; v++) {
                int weight = graph->adj[u][v];
                if (weight < INF) {
                    if (dist[u] + weight < dist[v]) { // if we can relax one more step, then we find a negative cycle
                        has_change = 1;;
                    }
                }
            }
        }
        has_negative_cycle = has_change;
    }

    if (has_negative_cycle) {
        save_negative(filename);
    } else {
        save_dist_array(dist, n, filename);
    }

    double end_time = omp_get_wtime();
    *time = end_time - init_time;
}

void bellman_ford_locks(Graph *graph, int src, int p, char *filename, double *time) {
    
    int V = graph->V;
    int dist[V];

    int local_has_change[p];
    
    double init_time = omp_get_wtime();

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
    for (int i = 1; i <= V - 1; i++) {
        #pragma omp for schedule(static, V*V / omp_get_num_threads()) collapse(2)
        for (int j = 0; j < V; j++) {
            for(int k = 0; k < V; k++) {
                int weight = graph->adj[j][k];
                if ((dist[j] != INF) && 
                    (dist[j] + weight < dist[k])
                ) 
                {
                    omp_set_lock(&locks[k]);
                    dist[k] = dist[j] + weight;
                    omp_unset_lock(&locks[k]);
                    local_has_change[omp_get_thread_num()] = 1;
                }
            }
        }


        int has_change = 0;
        for (int rank = 0; rank < p; rank++) {
            has_change |= local_has_change[rank];
        }
        if (!has_change) {
            break;
        }
        
    }
    

    // Step 3: check for negative-weight cycles.  The above
    // step guarantees shortest distances if graph doesn't
    // contain negative weight cycle.  If we get a shorter
    // path, then there is a cycle.
    int negative = 0;
    for (int i = 0; i < V; i++) {
        #pragma omp parallel for schedule(static)
        for(int j = 0; j<V; j++) {
            int weight = graph->adj[i][j];
            if (dist[i] != INF
                && dist[i] + weight < dist[j]) {
                #pragma omp atomic
                    negative++;
            }
        }
    }
    
    // Step 4: destroy locks
    #pragma omp parallel for
    for (int i = 0; i < V; i++) {
        omp_destroy_lock(&locks[i]);
    }

    double end_time = omp_get_wtime();

    if (negative) {
        save_negative(filename);
        *time = end_time - init_time;
        return;
    }
    else {
        save_dist_array(dist, V, filename);
        *time = end_time - init_time;
    }
    
}

int swapArray(int *a, int *b, int n){
    int true_idx = 0;
    for (int i = 0; i < n; i++ )
    {
        if (!(b[i] == -1)) {
            a[true_idx] = b[i];
            true_idx++;
        }
        b[i] = -1;
    }
    return true_idx;
}

void bellman_ford_frontier(Graph *graph, int src, int p, char *filename, double *time) {
    
    int n = graph->V; //Vertices

    int negative_cycle = 0;

    int frontier_size = 1; int next_frontier_size = 0; int idx = 0; 

    int dist[n], frontier[n], next_frontier[n], has_changed[n];

    int local_start[p], local_end[p];

    double init_time = omp_get_wtime();

    //step 2: find local task range
    int ave = n / p;
    #pragma omp parallel for
    for (int i = 0; i < p; i++) {
        local_start[i] = ave * i;
        local_end[i] = ave * (i + 1);
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
    
    while (frontier_size > 0 && idx++ < n) {   
        // print actual frontier
        /*
        printf("Frontier %d\n", idx);
        for (int l=0; l<n;l++) printf("%d  ", frontier[l]);
        printf("\n");
        */
        #pragma omp parallel default(shared) private(next_frontier_size)  
        {
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
                    #pragma omp barrier
                }
            }
        }
        if (idx == n) {
            for (int i = 0; i < n; i++) {
                if (next_frontier[i] != -1) {
                   negative_cycle = 1;
                }
            }
        } 
        /*
        printf("End of the %d relaxation\n", idx);
        printf("Next frontier\n");
        for (int l=0; l<n;l++) printf("%d  ", next_frontier[l]);
        printf("\n");
        
        printf("Distances array");
        for (int l=0; l<n;l++) printf("%d  ", dist[l]);
        printf("\n");
        */
    
        // swap front1 and front2
        frontier_size = swapArray(frontier, next_frontier, n);
        
        //printf("Frontier size %d\n", frontier_size);
        
        for(int k=0; k<n; k++) has_changed[k]=0;
    }
    

    double end_time = omp_get_wtime();

    *time = end_time - init_time;

    if (negative_cycle) {
        save_negative(filename);
    } else {
        save_dist_array(dist, n, filename);
    }
    
}