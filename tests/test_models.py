import os
import tempfile

import torch
import numpy as np
import pytorch_lightning as pl

from truthful_counterfactuals.models import MockModel
from truthful_counterfactuals.models import EnsembleModel
from truthful_counterfactuals.models import GINModel
from truthful_counterfactuals.testing import get_mock_graphs
from truthful_counterfactuals.data import loader_from_graphs


class TestMockModel():
    
    def test_saving_loading_works(self):
        """
        The MockModel should be able to save and load its state dictionary to and from a file.
        """
        node_dim = 10
        edge_dim = 3
        
        model = MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1)
        state_initial = model.state_dict()
        
        with tempfile.TemporaryDirectory() as path:
            model_path = os.path.join(path, 'model.ckpt')
            model.save(model_path)
            
            model = MockModel.load(model_path)
            assert isinstance(model, MockModel)
        
        state_loaded = model.state_dict()
        assert state_initial.keys() == state_loaded.keys()
        
        for key in state_initial.keys():
            assert torch.allclose(state_initial[key], state_loaded[key])


class TestEnsembleModel():
    
    def test_construction_basically_works(self):
        """
        It should be possible to construct a new instance of the ensemble model by passing it a list 
        of other models in the constructor.
        """
        node_dim = 10
        edge_dim = 3
        
        models = [MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1) for _ in range(5)]
        ensemble = EnsembleModel(models=models)
        
        assert isinstance(ensemble, EnsembleModel)
        assert len(ensemble.models) == 5
        
    def test_model_forward_works(self):
        """
        It should be possible to perform a forward pass with the ensemble model as with any other 
        AbstractGraphModel and it should return the mean prediction of the ensemble members.
        """
        node_dim = 10
        edge_dim = 3
        
        models = [MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1) for _ in range(5)]
        ensemble = EnsembleModel(models=models)
        
        graphs = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        loader = loader_from_graphs(graphs)
        data = next(iter(loader))
        
        info = ensemble(data)
        assert isinstance(info, dict)
        assert len(info) != 0
        assert 'graph_output' in info
        assert isinstance(info['graph_output'], torch.Tensor)
        
    def test_model_forward_graphs_works(self):
        """
        If the forward method itself works, it should also be possible to use the forward_graphs
        wrapper to directly perform predictions on graph dicts.
        """
        node_dim = 10
        edge_dim = 3
        
        models = [MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1) for _ in range(5)]
        ensemble = EnsembleModel(models=models)
        
        graphs = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        
        infos = ensemble.forward_graphs(graphs)
        assert isinstance(infos, list)
        assert len(infos) == 10
        
        for info in infos:
            assert isinstance(info, dict)
            assert 'graph_output' in info
            assert isinstance(info['graph_output'], np.ndarray)
            
    def test_saving_loading_works(self):
        """
        Like any other subclass of AbstractGraphModel, the EnsembleModel should be able to save and
        load its state dictionary to and from a absolute path.
        """
        node_dim = 10
        edge_dim = 3
        
        models = [MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1) for _ in range(5)]
        ensemble = EnsembleModel(models=models)
        
        graphs = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        data = next(iter(loader_from_graphs(graphs)))
        info_original = ensemble.forward(data)
        
        with tempfile.TemporaryDirectory() as path:
            model_path = os.path.join(path, 'model.ckpt')
            ensemble.save(model_path)
            
            ensemble = EnsembleModel.load(model_path)
            assert isinstance(ensemble, EnsembleModel)
            assert len(ensemble.models) == 5
            
            info_loaded = ensemble.forward(data)
            
        # The predictions of the ensemble model should be the same after saving and loading!
        for key, value in info_original.items():
            assert torch.allclose(info_original[key], info_loaded[key])


class TestGINModel():
    
    def test_construction_basically_works(self,):
        """
        It should be possible to construct a new instance of the model and the number of 
        layers should be determined by the unit lists given to the constructor.
        """
        model = GINModel(
            node_dim=10,
            edge_dim=3,
            encoder_units=[64, 64, 64],
            predictor_units=[32, 1],
            mode='regression',
        )
        assert model is not None
        assert len(model.encoder_layers) == 3
        assert len(model.predictor_layers) == 2
        
    def test_model_forward_works(self,):
        """
        It should be possible to excute a model forward pass with a batch of data without 
        causing any errors.
        """
        node_dim = 10
        edge_dim = 3
        
        model = GINModel(
            node_dim=node_dim,
            edge_dim=edge_dim,
            encoder_units=[64, 64, 64],
            predictor_units=[32, 1],
            mode='regression',
        )
        
        graphs = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        loader = loader_from_graphs(graphs)
        
        for data in loader:
            info = model(data)
            assert isinstance(info, dict)
            assert len(info) != 0
            assert 'graph_output' in info
            
    def test_training_works(self):
        """
        Since the GINModel inherits from pytorch_lightning.LightningModule it should be possible
        to train the model using the pytorch_lightning.Trainer class for a few epochs and then 
        it should be verifyable that the model parameters have actually been changed.
        """
        node_dim = 10
        edge_dim = 3
        
        model = GINModel(
            node_dim=node_dim,
            edge_dim=edge_dim,
            encoder_units=[64, 64, 64],
            predictor_units=[32, 1],
            mode='regression',
        )
        state_initial = model.state_dict()
        
        graphs = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        loader = loader_from_graphs(graphs)
        
        trainer = pl.Trainer(
            max_epochs=5,
        )
        trainer.fit(
            model,
            train_dataloaders=loader,
        )
        state_trained = model.state_dict()
        
        # Now we want to make certain that the model parameters have actually been changed 
        # during the training!
        assert not all([torch.equal(state_initial[key], state_trained[key]) for key in state_initial.keys()])
        
    def test_mean_variance_mixin_basically_works(self):
        """
        The GINModel implements the MeanVarianceMixin which should add the capability to
        predict the variance of the model output and to train the model with the mean-variance
        loss function.
        """
        node_dim = 10
        edge_dim = 3
        
        model = GINModel(
            node_dim=node_dim,
            edge_dim=edge_dim,
            encoder_units=[64, 64, 64],
            predictor_units=[32, 1],
            mode='regression',
            variance_units=[32, 16, 1],
        )
        # First of all we can check if the variance network was correctly initialized through the 
        # constructor argument "variance_units".
        assert len(model.variance_layers) == 3
        
        graphs = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        loader = loader_from_graphs(graphs)
        data = next(iter(loader))

        info = model.forward(data)        
        assert 'graph_embedding' in info
        # The result of this method should be the variance vector that was obtained from the 
        # variance subnetwork...
        variance = model.predict_variance(info['graph_embedding'])
        assert isinstance(variance, torch.Tensor)
        assert variance.shape == (10, 1)
        
        # ~ variance training
        model.enable_variance_training()
        
        trainer = pl.Trainer(max_epochs=5)
        trainer.fit(model, train_dataloaders=loader)
        print(trainer.logged_metrics)
        assert 'mve_loss' in trainer.logged_metrics
        assert not torch.any(torch.isnan(trainer.logged_metrics['mve_loss']))
        
    def test_saving_loading_works(self):
        """
        The model.save(path) method should save the model as a checkpoint file to the given path 
        and the load(path) class method should load that persistent model back into memory.
        After saving and loading a model it should give the same predictions when given the same input.
        """
        node_dim = 10
        edge_dim = 3
        
        model = GINModel(
            node_dim=node_dim,
            edge_dim=edge_dim,
            encoder_units=[64, 64, 64],
            predictor_units=[32, 1],
            mode='regression',
        )
        model.eval()
        
        graphs = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        data = next(iter(loader_from_graphs(graphs)))
        info_original = model.forward(data)
        
        # ~ saving and loading the model
        with tempfile.TemporaryDirectory() as path:
            model_path = os.path.join(path, 'model.ckpt')
            model.save(model_path)
            
            model = GINModel.load(model_path)
        
        # When loadign the model, the predictions of the model should still be the same
        info_loaded = model.forward(data)
        assert torch.allclose(info_original['graph_output'], info_loaded['graph_output'])
        assert torch.allclose(info_original['graph_embedding'], info_loaded['graph_embedding'])