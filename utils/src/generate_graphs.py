import os
import random 
import argparse


def generate_graph(vertexes: int, 
                   file_name: str, 
                   edges_density = 1,
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
    edges = [(i, j, random.randint(-100, -1)) 
                            if (vertexes*i + j) < neg_perc*(vertexes**2)
                            else (i, j, random.randint(1, 100)) 
                            for i in range(vertexes) for j in range(vertexes)
    ]

    edges = random.sample(edges, k = random.randint(len(edges)*edges_density, len(edges)))

    # Write the graph to a file

    cur_path = os.getcwd().split('utils')[0]

    if not os.path.exists(os.path.join(cur_path, 'tests')):
        os.makedirs(os.path.join(cur_path, 'tests', 'graphs'))
    elif not os.path.exists(os.path.join(cur_path, 'tests', 'graphs')):
        os.makedirs(os.path.join(cur_path, 'tests', 'graphs'))
    


    graph_path = os.path.join(cur_path, 'tests', 'graphs')

    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    
    with open(os.path.join(graph_path, f'{file_name}.txt'), 'w') as f:
        f.write(f'{vertexes}\n')
        f.write(f'{len(edges)}\n')
        for edge in edges:
            f.write(f'{edge[0]} {edge[1]} {edge[2]}\n')

def generate_graphs(n_graphs, 
                    size, 
                    edges_perc,
                    neg_perc)-> None:
    '''
    Generate and save n_graphs with size vertexes in a file.
    '''
    random.seed(42) # Set seed for reproducibility

    print(f'Generating {n_graphs} graphs of size {size}')
    
    figs = len(str(n_graphs))
    for i in range(n_graphs):

        idx = "0"*(figs - len(str(i))) + f"{i}"

        generate_graph(size, 
                       f'{size}_{idx}',
                       edges_perc,
                       neg_perc)
    print('Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate a graph')
    parser.add_argument('-s', type=int, help='The number of vertexes in the graph', default=100)
    parser.add_argument('-n', type=int, help='The number of graphs to generate', default=1)
    parser.add_argument('--edges_density', type=float, default=1.0)
    parser.add_argument('--negative_edges', type=float, default=.05)

    args = parser.parse_args()

    generate_graphs(args.n, 
                    args.s, 
                    edges_perc = args.edges_density, 
                    neg_perc = args.negative_edges)