import os
import random 
import argparse
from graphviz import Digraph


def generate_graph(n_vertexes: int, 
                   file_name: str, 
                   edges_perc = 1,
                   neg_perc = .05) -> None:
    '''
    A graph is just a set of edges, where each edge has just three int values.
    The first two values are the vertexes that the edge connects, and the third
    value is the weight of the edge (which can be negative).

    :param n_vertexes     : The number of vertexes in the graph
    :param file_name      : The name of the file where the graph will be written
    :param edges_perc     : Float from 0 to 1 indicating how much the graph 
                            must be connected, if 1 then edges = n_vertexes^2    

    '''

    edges = [(i, j, random.randint(-100*neg_perc, 100)) 
                        for i in range(n_vertexes) 
                        for j in range(n_vertexes) 
                        if i != j
    ]
    edges = random.sample(edges, k = random.randint(len(edges)*edges_perc , len(edges)))

    # Write the graph to a file

    cur_path = os.getcwd().split('src')[0]
    graph_path = os.path.join(cur_path, 'tests', 'graphs')

    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    
    with open(os.path.join(graph_path, f'{file_name}.txt'), 'w') as f:
        f.write(f'{n_vertexes}\n')
        f.write(f'{len(edges)}\n')
        for edge in edges:
            f.write(f'{edge[0]} {edge[1]} {edge[2]}\n')



def save_graph(filename, view = False):
    folder = os.getcwd().split("src")[0]

    path = os.path.join(folder, 'tests', 'graphs', filename)
    
    # Read file and create graph
    with open(path, 'r') as f:
        lines = f.readlines()

    dot = Digraph(comment='Graph', format='png')

    nodes = []
    for i in range(2, len(lines)):
        line = lines[i].strip().split(' ')
        if line[0] not in nodes:
            nodes.append(line[0])
            dot.node(line[0])
        if line[1] not in nodes:
            nodes.append(line[1])
            dot.node(line[1])

        dot.edge(line[0], line[1], label=line[2])
    
    # Save graph
    dot.render(os.path.join(folder, 'tests', 'imgs', filename.split('.')[0]), view=view)

def generate_graphs(n_graphs, 
                    size, 
                    viz,
                    edges_perc,
                    neg_perc)-> None:
    '''
    Generate and save n_graphs with size vertexws in a file.
    '''
    random.seed(42) # Set seed for reproducibility

    print(f'Generating {n_graphs} graphs of size {size}')

    for i in range(n_graphs):
        generate_graph(size, 
                       f'{size}_{i}',
                       edges_perc,
                       neg_perc)

        if viz:
            save_graph(f'{size}_{i}.txt', view = False)
    
    print('Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate a graph')
    parser.add_argument('size', type=int, help='The number of vertexes in the graph', default=100)
    parser.add_argument('n_graphs', type=int, help='The number of graphs to generate', default=1)
    parser.add_argument('--viz', type=int, help="Save also a visualization of the graph", default=0)
    parser.add_argument('--edges_perc', type=float, default=1.0)
    parser.add_argument('--neg_perc', type=float, default=.05)

    args = parser.parse_args()

    generate_graphs(args.n_graphs, 
                    args.size, 
                    viz=bool(args.viz), 
                    edges_perc = args.edges_perc, 
                    neg_perc = args.neg_perc)
    