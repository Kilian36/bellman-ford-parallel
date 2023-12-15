#include "utils.h"


/*Utilities for graph in the form struct AdjGraph*/
void print_graph(Graph graph){
    printf("Graph with %d vertices\n", graph.V);
    for(int i = 0; i < graph.V; i++){
        for(int j = 0; j < graph.V; j++){
            printf("%d\t", graph.adj[i][j]);
        }
        printf("\n");
        printf("\n");
    } 
}

struct Graph read_graph(char* filename)
{
    FILE * fp;
    char * line = NULL;
    size_t len = 0;

    int V;

    fp = fopen(filename, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    printf("File opened\n");
    getline(&line, &len, fp);
    V = atoi((const char *) line);
    
    getline(&line, &len, fp);

    int **adj = (int**)malloc(V * sizeof(int *));

    for (int i=0; i<V; i++) {
        adj[i] = (int *)malloc(V * sizeof(int));
        if (adj[i] == NULL) {
            // Handle memory allocation failure
            exit(EXIT_FAILURE);
        }
    }

    // Initialize the graph
    for (int i=0; i<V; i++) {
        for (int j=0; j<V; j++) {
            adj[i][j] = 0;
        }
    }
    
    // Declare a bidimensional array of size VxV and initialize it to 0 statically
    while ((getline(&line, &len, fp)) != -1) {
        
        char *scr, *dst, *w;
        scr = strtok(line, " ");
        dst = strtok(NULL, " ");
        w = strtok(NULL, " ");

        
        int source = atoi((const char *) scr);
        int destination = atoi((const char *) dst);
        int my_weight = atoi((const char *) w);
        
        adj[source][destination] = my_weight;
    }

    fclose(fp);
    if (line)
        free(line);
    
    return (struct Graph) {V, adj};
}   

void free_graph(Graph *graph){
    for(int i = 0; i < graph->V; i++){
        free(graph->adj[i]);
    }
    free(graph->adj);
}

struct Graph1D read_graph1D(char* filename)
{
    FILE * fp;
    char * line = NULL;
    size_t len = 0;

    int V;

    fp = fopen(filename, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    printf("File opened\n");
    getline(&line, &len, fp);
    V = atoi((const char *) line);
    
    getline(&line, &len, fp);

    int *adj = (int*)malloc(V * V * sizeof(int *));

    // Initialize the graph
    for (int i=0; i<V; i++) {
        for (int j=0; j<V; j++) {
            adj[i*V + j] = 0;
        }
    }
    
    // Declare a bidimensional array of size VxV and initialize it to 0 statically
    while ((getline(&line, &len, fp)) != -1) {
        
        char *scr, *dst, *w;
        scr = strtok(line, " ");
        dst = strtok(NULL, " ");
        w = strtok(NULL, " ");

        int source = atoi((const char *) scr);
        int destination = atoi((const char *) dst);
        int my_weight = atoi((const char *) w);
        
        adj[source*V + destination] = my_weight;
    }

    fclose(fp);
    if (line)
        free(line);
    
    return (struct Graph1D) {V, adj};
}   

void save_dist_array(int *dist, int n, char *filename){
    FILE *fp;
    // We write in the first line the original node
    fp = fopen(filename, "w");
    printf("Saving to file %s\n", filename);
    printf("\n");

    if(fp != NULL) {
        for(int i=0; i<n; i++) {
            if (dist[i] == INF) fprintf(fp, "%d inf\n", i);
            else fprintf(fp, "%d %d\n", i, dist[i]);
        }

    }
    fclose(fp);
}

void save_negative(char *filename) {
    /*
        Negative cylces graphs are indicated with -1 -1
    */
    FILE *fp;

    fp = fopen(filename, "w");
    printf("Saving to file %s\n", filename);
    printf("\n");

    if (fp != NULL) fprintf(fp, "%d %d\n", -1, -1);
    fclose(fp);

}

void print_dist_array(int *dist, int n)
{
    printf("Vertex  Distance from Source\n");
    for (int i = 0; i < n; ++i)
        printf("%d \t\t %d\n", i, dist[i]);
}