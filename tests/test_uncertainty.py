import pytest
import random
from typing import List

import torch
import numpy as np

from truthful_counterfactuals.models import MockModel
from truthful_counterfactuals.models import EnsembleModel
from truthful_counterfactuals.uncertainty import EnsembleUncertainty
from truthful_counterfactuals.uncertainty import RandomUncertainty
from truthful_counterfactuals.uncertainty import EnsembleMveUncertainty
from truthful_counterfactuals.uncertainty import EnsembleGradientUncertainty
from truthful_counterfactuals.uncertainty import TrustScoreUncertainty
from truthful_counterfactuals.testing import get_mock_graphs
from truthful_counterfactuals.data import loader_from_graphs


class TestRandomUncertainty():
    
    def test_basically_works(self):
        
        node_dim = 10
        edge_dim = 3
        
        model = MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1)
        uncertainty_estimator = RandomUncertainty(model)
        
        graphs = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        results: list[dict] = uncertainty_estimator.evaluate_graphs(graphs)
        
        for result in results:
            assert 'uncertainty' in result
            assert 'prediction' in result
            
    def test_calibrate_works(self):
        
        node_dim = 10
        edge_dim = 3
        
        model = MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1)
        uncertainty_estimator = RandomUncertainty(model)
        
        graphs = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        results: list[dict] = uncertainty_estimator.evaluate_graphs(graphs)
        
        errors = [random.uniform(0.1, 0.2) for graph in graphs]
        ucs_cal, models_cal = uncertainty_estimator.calibrate(graphs, errors)
        assert isinstance(ucs_cal, np.ndarray)
        assert isinstance(models_cal, dict)
        assert len(models_cal) == 1
        
        results_cal = uncertainty_estimator.evaluate_graphs(graphs)
        
        ucs = np.array([result['uncertainty'] for result in results])
        ucs_cal = np.array([result['uncertainty'] for result in results_cal])
        
        # In the end the whole point is that the calibrated uncertainty values should be something 
        # else than the original ones and more specifically on the same set they should be limited 
        # by the min and max values of the given calibration errors.
        assert not np.allclose(ucs, ucs_cal)
        assert np.all(np.min(ucs_cal) >= np.min(errors))
        assert np.all(np.max(ucs_cal) <= np.max(errors))


class TestEnsembleUncertainty():
    
    def test_basically_works(self):
        """
        It should be possible to construct a new instance of the ensemble uncertainty with a list 
        of models and then use that to evaluat the uncertainty on a list of graphs.
        """
        node_dim = 10
        edge_dim = 3
        
        # We use untrained mock models here as the basis of the ensemble because the actual model 
        # accuracy is not important for the basic funcitonality test.
        models = [MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1) for _ in range(5)]
        ensemble = EnsembleModel(models)
        uncertainty_estimator = EnsembleUncertainty(ensemble)
        
        graphs = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        results: list[dict] = uncertainty_estimator.evaluate_graphs(graphs)
        assert len(results) == len(graphs)
        
        for result in results:
            assert 'uncertainty' in result
            assert 'prediction' in result
            

class TestEnsembleMveUncertainty():
    
    def test_basically_works(self):
        
        node_dim = 10
        edge_dim = 3
        
        models = [MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1) for _ in range(5)]
        ensemble = EnsembleModel(models)
        
        uncertainty_estimator = EnsembleMveUncertainty(ensemble)
        graphs = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        results: list[dict] = uncertainty_estimator.evaluate_graphs(graphs)
        assert len(results) == len(graphs)
        
        for result in results:
            assert 'uncertainty' in result
            assert 'prediction' in result
            assert '_ens_uncertainty' in result
            assert '_mve_uncertainty' in result
            
    def test_calibrate_works(self):
        """
        Since this is a composite uncertainty estimator that combines the ensemble uncertainty and
        the MVE uncertainty, the calibration should be applied to both of these components and is therefore
        a custom implementation.
        """
        node_dim = 10
        edge_dim = 3
        
        models = [MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1) for _ in range(5)]
        ensemble = EnsembleModel(models)
        
        uncertainty_estimator = EnsembleMveUncertainty(ensemble)
        
        graphs = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        results: list[dict] = uncertainty_estimator.evaluate_graphs(graphs)
        ucs = np.array([result['uncertainty'] for result in results])
        
        errors = [random.uniform(0.1, 0.2) for graph in graphs]
        ucs_cal, models_cal = uncertainty_estimator.calibrate(graphs, errors)
        assert isinstance(ucs_cal, np.ndarray)
        assert isinstance(models_cal, dict)
        assert models_cal
        
        results_cal = uncertainty_estimator.evaluate_graphs(graphs, calibrated=True)
        ucs_cal = np.array([result['uncertainty'] for result in results_cal])
        
        # In the end the whole point is that the calibrated uncertainty values should be something 
        # else than the original ones and more specifically on the same set they should be limited 
        # by the min and max values of the given calibration errors.
        assert not np.allclose(ucs, ucs_cal)
        assert np.all(np.min(ucs_cal) >= np.min(errors))
        assert np.all(np.max(ucs_cal) <= np.max(errors))
        
        
class TestEnsembleGradientUncertainty:
    
    def test_gnn_gradient_works(self):
        
        graphs = get_mock_graphs(10, node_dim=10, edge_dim=3)
        data = next(iter(loader_from_graphs(graphs)))
        data.x.requires_grad = True
        
        model = MockModel(node_dim=10, edge_dim=3, out_dim=1)
        info = model(data)
        
        grad_outputs = torch.ones_like(info['graph_output'])
        grad = torch.autograd.grad(
            outputs=info['graph_output'],
            inputs=data.x,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True,
        )
        print(type(grad), grad)
        print(data.x.shape)
        print(grad[0].shape)
        
    def test_basically_works(self):
        node_dim = 10
        edge_dim = 3
        
        models = [MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1) for _ in range(5)]
        ensemble = EnsembleModel(models)
        
        uncertainty_estimator = EnsembleGradientUncertainty(ensemble)
        
        graphs = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        results: list[dict] = uncertainty_estimator.evaluate_graphs(graphs)
        ucs = np.array([result['uncertainty'] for result in results])
        
        errors = [random.uniform(0.1, 0.2) for graph in graphs]
        ucs_cal, models_cal = uncertainty_estimator.calibrate(graphs, errors)
        assert isinstance(ucs_cal, np.ndarray)
        assert isinstance(models_cal, dict)
        assert models_cal


class TestTrustScoreUncertainty:
    
    def test_regression_basically_works(self):
        """
        If it is generally possible to construct the TrustScoreEstimator with a model and a set of 
        training graphs and targets, then it should also be possible to evaluate the uncertainty
        on a set of graphs.
        """
        node_dim = 10
        edge_dim = 3
        num_train = 20
        
        model = MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1)
        graphs_train = get_mock_graphs(num_train, node_dim=node_dim, edge_dim=edge_dim)
        targets_train = [random.uniform(0, 1) for _ in range(num_train)]
        
        # The TrustScoreEstimator should be able to be constructed with a model and a set of
        # training graphs and targets.
        uncertainty_estimator = TrustScoreUncertainty(
            model=model,
            graphs_train=graphs_train,
            targets_train=targets_train,
            mode='regression'
        )
        
        graphs = get_mock_graphs(5, node_dim=node_dim, edge_dim=edge_dim)
        for graph in graphs:
            graph['graph_output'] = random.uniform(0, 1)
        results: list[dict] = uncertainty_estimator.evaluate_graphs(graphs)
        
        # The result needs to contain the uncertainty and the prediction values and these 
        # need to be numerical values.
        assert len(results) == len(graphs)
        for result in results:
            
            assert 'uncertainty' in result
            assert isinstance(result['uncertainty'], (int, float))
            
            assert 'prediction' in result
            assert isinstance(result['prediction'], (int, float))
            
    def test_calibrate_works(self):
        node_dim = 10
        edge_dim = 3
        
        model = MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1)
        graphs_train = get_mock_graphs(10, node_dim=node_dim, edge_dim=edge_dim)
        targets_train = [random.uniform(0, 1) for _ in range(10)]
        
        uncertainty_estimator = TrustScoreUncertainty(
            model=model,
            graphs_train=graphs_train,
            targets_train=targets_train,
            mode='regression'
        )
        
        graphs = get_mock_graphs(5, node_dim=node_dim, edge_dim=edge_dim)
        for graph in graphs:
            graph['graph_output'] = random.uniform(0, 1)
        results: list[dict] = uncertainty_estimator.evaluate_graphs(graphs)
        
        errors = [random.uniform(0.1, 0.2) for graph in graphs]
        ucs_cal, models_cal = uncertainty_estimator.calibrate(graphs, errors)
        assert isinstance(ucs_cal, np.ndarray)
        assert isinstance(models_cal, dict)
        assert len(models_cal) == 1
        
        results_cal = uncertainty_estimator.evaluate_graphs(graphs)
        
        ucs = np.array([result['uncertainty'] for result in results])
        ucs_cal = np.array([result['uncertainty'] for result in results_cal])
        
        assert not np.allclose(ucs, ucs_cal)
        assert np.all(np.min(ucs_cal) >= np.min(errors))
        assert np.all(np.max(ucs_cal) <= np.max(errors))
        
    def test_classification_basically_works(self):
        """
        If it is generally possible to construct the TrustScoreEstimator with a model and a set of 
        training graphs and targets, then it should also be possible to evaluate the uncertainty
        on a set of graphs in the classification mode.
        """
        node_dim = 10
        edge_dim = 3
        num_train = 20
        
        model = MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=2)
        graphs_train: List[dict] = get_mock_graphs(num_train, node_dim=node_dim, edge_dim=edge_dim)
        targets_train: np.ndarray = np.array([random.choice([[0, 1], [1, 0]]) for _ in range(num_train)])
        
        # The TrustScoreEstimator should be able to be constructed with a model and a set of
        # training graphs and targets.
        uncertainty_estimator = TrustScoreUncertainty(
            model=model,
            graphs_train=graphs_train,
            targets_train=targets_train,
            mode='classification'
        )
        
        graphs = get_mock_graphs(5, node_dim=node_dim, edge_dim=edge_dim)
        for graph in graphs:
            graph['graph_output'] = np.array(random.choice([[0, 1], [1, 0]]))
            
        results: list[dict] = uncertainty_estimator.evaluate_graphs(graphs)
        
        # The result needs to contain the uncertainty and the prediction values and these 
        # need to be numerical values.
        assert len(results) == len(graphs)
        for result in results:
            
            assert 'uncertainty' in result
            assert isinstance(result['uncertainty'], (int, float, np.ndarray))
            
            assert 'prediction' in result
            assert isinstance(result['prediction'], (int, float, np.ndarray))
            
    def test_tanimoto_distance_metric_works(self):
        """
        If it is generally possible to construct the TrustScoreEstimator with a model and a set of 
        training graphs and targets, then it should also be possible to evaluate the uncertainty
        on a set of graphs using the "tanimoto" distance metric.
        """
        node_dim = 10
        edge_dim = 3
        num_train = 20
        
        model = MockModel(node_dim=node_dim, edge_dim=edge_dim, out_dim=1)
        graphs_train = get_mock_graphs(num_train, node_dim=node_dim, edge_dim=edge_dim)
        targets_train = np.array([random.uniform(0, 1) for _ in range(num_train)])
        
        # Add the "graph_repr" property with random SMILES strings
        for graph in graphs_train:
            graph['graph_repr'] = random.choice(['CCO', 'CCN', 'CCC', 'CCCl', 'CCBr'])
        
        # The TrustScoreEstimator should be able to be constructed with a model and a set of
        # training graphs and targets.
        uncertainty_estimator = TrustScoreUncertainty(
            model=model,
            graphs_train=graphs_train,
            targets_train=targets_train,
            distance_metric='tanimoto',
            mode='regression'
        )
        
        graphs = get_mock_graphs(5, node_dim=node_dim, edge_dim=edge_dim)
        for graph in graphs:
            graph['graph_repr'] = random.choice(['CCO', 'CCN', 'CCC', 'CCCl', 'CCBr'])
            graph['graph_output'] = random.uniform(0, 1)
        results: list[dict] = uncertainty_estimator.evaluate_graphs(graphs)
        
        # The result needs to contain the uncertainty and the prediction values and these 
        # need to be numerical values.
        assert len(results) == len(graphs)
        for result in results:
            
            assert 'uncertainty' in result
            assert isinstance(result['uncertainty'], (int, float, np.ndarray))
            
            assert 'prediction' in result
            assert isinstance(result['prediction'], (int, float, np.ndarray))