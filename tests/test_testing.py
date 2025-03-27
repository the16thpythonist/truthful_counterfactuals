from visual_graph_datasets.typing import assert_graph_dict
from truthful_counterfactuals.testing import get_mock_graphs


def test_get_mock_graphs():
    
    graphs = get_mock_graphs(10)
    assert len(graphs) == 10
    
    for graph in graphs:
        assert_graph_dict(graph)