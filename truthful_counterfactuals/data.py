import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def data_from_graph(graph: dict) -> Data:
    
    data = Data(
        x=torch.tensor(graph['node_attributes'], dtype=torch.float),
        edge_index=torch.tensor(graph['edge_indices'].T.astype(int), dtype=torch.int64),
        edge_attr=torch.tensor(graph['edge_attributes'], dtype=torch.float),
    )
    
    if 'graph_labels' in graph:
        data.y = torch.tensor(graph['graph_labels'], dtype=torch.float)
        
    return data


def data_list_from_graphs(graphs: dict) -> list[Data]:
    
    return [data_from_graph(graph) for graph in graphs]


def loader_from_graphs(graphs: dict, batch_size: int = -1) -> DataLoader:
    
    if batch_size == -1:
        batch_size = len(graphs)
    
    data_list = data_list_from_graphs(graphs)
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    
    return loader