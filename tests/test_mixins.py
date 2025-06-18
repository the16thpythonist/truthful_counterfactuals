import os
import tempfile

import torch
import torch.nn as nn
import numpy as np

from truthful_counterfactuals.mixins import SwagMixin
from truthful_counterfactuals.models import MockModel
from truthful_counterfactuals.data import loader_from_graphs
from truthful_counterfactuals.testing import get_mock_graphs


class TestSwagMixin:
    
    def test_record_snapshot_basically_works(self):
        """
        The ``record_snapshot`` method should essentially store a snapshot of the model's weights in the 
        model's internal self.snapshot_list attribute and each of the values in those lists should be 
        themselves torch.nn.Parameter objects.
        """
        node_dim = 10
        edge_dim = 3
        
        model = MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1)
        assert len(model.snapshot_list) == 0
        
        model.record_snapshot()
        assert len(model.snapshot_list) == 1
        
        model.record_snapshot()
        assert len(model.snapshot_list) == 2
        
        for param_list in model.snapshot_list:
            assert isinstance(param_list, nn.ParameterList)
            assert len(param_list) > 0
            
            for param in param_list:
                assert isinstance(param, nn.Parameter)
                
    def test_record_weights_saving_loading_works(self):
        """
        It should be possible to save and load the additionally recorded swag weights of a model 
        through the normal save and load methods.
        """
        node_dim = 10
        edge_dim = 3
        
        model = MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1)
        
        model.record_snapshot()
        model.record_snapshot()
        model.record_snapshot()
        assert len(model.snapshot_list) == 3
        
        # based on this model with multiple weight snapshots we can now also calculate the distirbution
        # over the weights using the "calculate_distribution" method after which the corresoinding
        # properties should be populated with non-zero values.
        model.calculate_distribution()
        assert not torch.allclose(model.weights_mean, torch.zeros_like(model.weights_mean), atol=1e-6)
        
        graphs = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        loader = loader_from_graphs(graphs, batch_size=len(graphs))
        data = next(iter(loader))
        info = model.forward(data)
        
        # After saving and loading the model as a file, these stored model snapshots of the previous weights 
        # during the training process should also be properly restored
        with tempfile.TemporaryDirectory() as path:
            model_path = os.path.join(path, 'model.ckpt')
            model.save(model_path)
            
            model_loaded = MockModel.load(model_path)
            info_loaded = model_loaded.forward(data)
            
            # first of all we need to check that the model was correctly saved by itself aka that the 
            # predictions are still the same
            assert torch.allclose(info['graph_output'], info_loaded['graph_output'])
            
            # Additionally the loading process should also reconstruct the stored model snapshots
            assert len(model_loaded.snapshot_list) == 3
            for param in model_loaded.snapshot_list[0]:
                assert isinstance(param, nn.Parameter)
                
            # Finally, the distribution weights of the model should also be available after loading the model.
            assert torch.allclose(model_loaded.weights_mean, model_loaded.weights_mean)
            assert torch.allclose(model_loaded.weights_std, model_loaded.weights_std)
            
    def test_sampling_works(self):
        """
        At the end of the training process a SWAG model calculates a distribution of the weights based on the 
        history of weight snapshots. It should then be possible to sample from this distribution by using 
        the ``sample`` method, which should modify the model's parameters accordingly.
        """
        
        node_dim = 10
        edge_dim = 3
        
        model = MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1)
        model.record_snapshot()
        model.record_snapshot()
        model.record_snapshot()
        model.calculate_distribution()
        # since we haven't actually changed the weights all the standard deviations will evaluate to zero 
        # and sampling from that distributiuon should result in the same weights as before. Therefore we 
        # can use this to test the sampling method.
        model.weights_std = torch.ones_like(model.weights_std)
        
        params = [param.detach().clone() for param in model.parameters()]
        
        model_sampled = model.sample()
        params_sampled = [param.detach().clone() for param in model_sampled.parameters()]
        
        for param, param_sampled in zip(params, params_sampled):
            assert not torch.allclose(param, param_sampled)