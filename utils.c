#include "utils.h"

// Utility function, to print a graph
void print_graph(struct Graph graph){
    printf("Graph with %d vertices and %d edges\n", graph.V, graph.E);
    for(int i = 0; i < graph.E; i++){
        printf("Edge %d  : %d -> %d\n", i, graph.edge[i].src, graph.edge[i].dest);
        printf("Weight :      %d\n", graph.edge[i].weight);
    } 
}
struct Graph create_graph(int n_vert, int n_edges) {
    struct Edge* edges = malloc(n_edges * sizeof(struct Edge));
    if (edges == NULL) {
        // Handle memory allocation failure
        exit(EXIT_FAILURE);
    }

    struct Graph graph = {n_vert, n_edges, edges};
    return graph;
}

void free_graph(struct Graph* graph) {
    free(graph->edge); 
}


struct Graph read_graph(char* filename)
{
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    int read;

    int V, E;

    fp = fopen(filename, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    printf("File opened\n");
    read = getline(&line, &len, fp);
    V = atoi((const char *) line);
    
    read = getline(&line, &len, fp);
    E = atoi((const char *) line);
    
    struct Graph graph = create_graph(V, E);
    
    int i = 0;
    while ((getline(&line, &len, fp)) != -1 && (i < E)) {
        
        char *scr, *dst, *w;
        scr = strtok(line, " ");
        dst = strtok(NULL, " ");
        w = strtok(NULL, " ");

        
        int source = atoi((const char *) scr);
        int destination = atoi((const char *) dst);
        int my_weight = atoi((const char *) w);
        
        graph.edge[i].src = source;
        graph.edge[i].dest = destination;
        graph.edge[i].weight = my_weight;
        
        i++;
        
    }

    fclose(fp);
    if (line)
        free(line);
    
    return graph;
}

void save_dist_array(int *dist, int n, char *filename){
    FILE *fp;
    // We write in the first line the original node
    fp = fopen(filename, "w");
    printf("Saving to file %s\n", filename);

    if(fp != NULL) {
        for(int i=0; i<n; i++) {
            if (dist[i] == INT_MAX)  fprintf(fp, "%d inf\n", i);
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

    if (fp != NULL) fprintf(fp, "%d %d\n", -1, -1);
    fclose(fp);

}

void print_dist_array(int *dist, int n)
{
    printf("Vertex  Distance from Source\n");
    for (int i = 0; i < n; ++i)
        printf("%d \t\t %d\n", i, dist[i]);
}