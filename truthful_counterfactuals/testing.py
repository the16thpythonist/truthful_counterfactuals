import random

import numpy as np
from torch_geometric.data import Data


def get_mock_graphs(num: int,
                    node_dim: int = 10,
                    edge_dim: int = 3,
                    out_dim: int = 1,
                    ) -> list[dict]:
    
    graph_list = []
    for index in range(num):
        
        num_nodes = random.randint(10, 20)
        node_indices = np.arange(num_nodes)
        node_attributes = np.random.rand(num_nodes, node_dim)
        
        # generate edge index tuples such that every node is connected with the next one
        # by cyclically shifting the node indices
        edge_indices = np.array([[i, (i + 1) % num_nodes] for i in range(num_nodes)], dtype=int)
        edge_attributes = np.random.rand(len(edge_indices), edge_dim)
        
        graph_labels = np.random.rand(out_dim)
        
        graph = {
            'node_indices':         node_indices,
            'node_attributes':      node_attributes,
            'edge_indices':         edge_indices,
            'edge_attributes':      edge_attributes,
            'graph_labels':         graph_labels,
        }
        graph_list.append(graph)
        
    return graph_list