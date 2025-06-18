import os
import typing as typ
from collections import defaultdict
from typing import Dict, List, Literal

import torch
import numpy as np
from rich.pretty import pprint
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors

from truthful_counterfactuals.mixins import MeanVarianceMixin
from truthful_counterfactuals.mixins import SwagMixin
from truthful_counterfactuals.mixins import EvidentialMixin
from truthful_counterfactuals.models import AbstractGraphModel
from truthful_counterfactuals.models import EnsembleModel
from truthful_counterfactuals.data import loader_from_graphs
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances


# == BASE CLASSES ==

class AbstractUncertainty():
    """
    Base class for ML uncertainty prediction.
    
    Sub-classes have to implement the ``evaluate_graphs`` method which takes a list of graphs and returns 
    a list of dictionaries with at least the following two keys:
    - "prediction": The prediction of the model for the corresponding input graph. This can usually directly 
      be the output of the model's forward pass. However, for some UQ methods, the prediction itself is also 
      calculated differently (such as the average of a model ensemble).
    - "uncertainty": The uncertainty value for the prediction.
    
    The constructor of the specific implementations can differ in their arguments. However, most of the time 
    they will need to include the model on which the uncertainty is to be evaluated in some way.
    """
    def __init__(self, mode: typ.Literal['regression', 'classification'] = 'regression'):
        self.mode = mode
        
        # We use this dictionary to track all of the calibration models that are created during the 
        # uncertainty calibration. In most cases we will only need a single calibration model so this 
        # dict will only have one element, but in the case of composite models we need multiple 
        # calibration models.
        # We'll always use the IsothonicRegression models for calibration.
        self.calibration_models: dict[str, IsotonicRegression] = {}
    
    def evaluate_graphs(self, graphs: list[dict], calibrated: bool = True) -> list[dict]:
        """
        This method needs to be implemented by the sub-classes. It takes a list of input ``graphs`` and 
        is supposed to return a list of dicts where each dict corresponds to one input element and is 
        supposed to contain at least the two keys "prediction" and "uncertainty".
        """
        raise NotImplementedError()
    
    def calibrate(self, graphs: list[dict], values: list[float]):
        """
        Can be used to calibrate the uncertainty values of an uncertainty estimator.
        
        **What is calibration?**
        
        Generally speaking, uncertainty quantification methods output uncertainty values that have a more 
        or less arbitrary scale. This means that the uncertainty values only have meaning relative to each 
        other but cannot be interpreted directly as a standard deviation / model errors.
        The process of calibration uses an external dataset (validation set) with known ground truth values 
        for the uncertainty (== model error) to map the predicted scale to the true scale therefore making 
        the uncertainty values interpretable / comparable.
        
        **Default Implementation**
        
        The default implementation of this method assumes that there is only a single uncertainty 
        quantification method that needs to be calibrated. Given the true uncertainty ``values`` for the 
        given ``graphs`` of an external validation set, the method fits an isotonic regression model to
        the predicted uncertainty values and the true uncertainty values.
        
        This default implementation can be overwritten for more complex calibration prcedures such as 
        for composite uncertainty quantification methods, where it is important to calibrate the individual
        methods separately to bring them to a common scale.
        
        :returns: a tuple, where the first element is the array of the calibrated uncertainty values on 
            the given validation set and the second element is the dictionary containing the calibration
            models.
        """
        results = self.evaluate_graphs(graphs, calibrated=False)
        
        uncertainty_pred = np.array([float(result['uncertainty']) for result in results])
        uncertainty_true = np.array([float(value) for value in values])
        
        self.calibration_models['main'] = IsotonicRegression(
            y_min=np.min(uncertainty_true) - 1, 
            y_max=np.max(uncertainty_true) + 1, 
            out_of_bounds='clip',
            increasing='auto',
        )
        
        uncertainty_cal = self.calibration_models['main'].fit_transform(uncertainty_pred, uncertainty_true)
        return uncertainty_cal, self.calibration_models
        
    
    
class MockUncertainty(AbstractUncertainty):
    """
    A mock implementation of the AbstractUncertainty base class that simply returns the given model's 
    predictions and random values for the uncertainty.
    """
    def __init__(self, 
                 model: AbstractGraphModel,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
    
    def evaluate_graphs(self, graphs: list[dict], calibrated: bool = True) -> list[dict]:
        
        infos = self.model.forward_graphs(graphs)
        
        results: list[dict] = []
        for info, graph in zip(infos, graphs):
            
            uncertainty = float(np.random.rand())
            if self.calibration_models and calibrated:
                uncertainty = self.calibration_models['main'].transform([np.random.rand()])[0]
            
            results.append({
                'prediction':       float(info['graph_output']),
                'uncertainty':      uncertainty,
            })
            
        return results    

    
# == SPECIFIC CLASSES ==
    
    
class RandomUncertainty(AbstractUncertainty):
    """
    A mock implementation of the AbstractUncertainty base class that simply returns the given model's
    predictions and random values for the uncertainty of each prediction.
    
    This is a reference implementation to compare others against.
    """
    def __init__(self, 
                 model: AbstractGraphModel,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.model = model
    
    def evaluate_graphs(self, graphs: list[dict], calibrated: bool = True) -> list[dict]:
        
        infos = self.model.forward_graphs(graphs)
        
        results: list[dict] = []
        for graph, info in zip(graphs, infos):
            
            uncertainty = float(np.random.rand())
            
            # optionally, we apply the internal calibration mapping to the raw uncertainty value
            if calibrated and 'main' in self.calibration_models:
                uncertainty = self.calibration_models['main'].transform([uncertainty])[0]
            
            results.append({
                'prediction':       float(info['graph_output']),
                'uncertainty':      uncertainty,
            })
            
        return results

    
class EnsembleUncertainty(AbstractUncertainty):
    """
    Estimates the uncertainty from the output standard deviation of an ensemble of models.
    
    Given an ``EnsembleModel`` instance, this class evaluates each input prediction as the average of 
    the individual model predictions and the uncertainty as the corresponding standard deviation.
    
    An ``EnsembleModel`` is simply a wrapper around a list of models that can be used to evaluate
    the same input data with each model and return the results in a list.
    """
    def __init__(self, 
                 ensemble: EnsembleModel,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        
        self.ensemble = ensemble
        self.models = ensemble.models
        
    def evaluate_graphs(self, graphs: list[dict], calibrated: bool = True) -> list[dict]:
        
        graph_predictions: dict[int, list[float]] = defaultdict(list)
        for model in self.models:
            infos = model.forward_graphs(graphs)
            for index, info in enumerate(infos):
                
                prediction = info['graph_output']
                
                if self.mode == 'classification':
                    prediction = np.argmax(prediction)
                
                graph_predictions[index].append(info['graph_output'])
                
        results: list[dict] = []
        for index, predictions in graph_predictions.items():
            
            uncertainty = np.std(predictions)
            
            # optionally, we apply the internal calibration mapping to the raw uncertainty value
            if calibrated and 'main' in self.calibration_models:
                uncertainty = self.calibration_models['main'].transform([uncertainty])[0]
            
            results.append({
                'avg':              np.mean(predictions),
                'std':              np.std(predictions),
                'prediction':       np.mean(predictions),
                'uncertainty':      uncertainty,
            })
            
        return results
    
    
class EnsembleGradientUncertainty(AbstractUncertainty):
    
    def __init__(self, 
                 ensemble: EnsembleModel,
                 aggregation_method: typ.Literal['mean', 'max'] = 'mean',
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        
        self.ensemble = ensemble
        self.models = ensemble.models
        self.aggregation_method = aggregation_method
        
    def gradients_graphs(self,
                         model: AbstractGraphModel,
                         graphs: list[dict],
                         ) -> list[dict]:
        
        data = next(iter(loader_from_graphs(graphs)))
        data.x.requires_grad = True
        
        info = model(data)
        grad_outputs = torch.ones_like(info['graph_output'])
        grad = torch.autograd.grad(
            outputs=info['graph_output'],
            inputs=data.x,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True,
        )
        
        infos = model.split_infos({'node_gradient': grad[0]}, data)
        return infos
    
    def aggregate_gradients(self, gradients: np.ndarray) -> np.ndarray:
        
        if self.aggregation_method == 'mean':
            return np.mean(gradients, axis=0)
        
        if self.aggregation_method == 'max':
            return np.max(gradients, axis=0)
        
    def evaluate_graphs(self, 
                        graphs: list[dict], 
                        calibrated: bool = True
                        ) -> list[dict]:
        
        # This data structure will map the graph index (key) to a list of predictions made 
        # be the different members of the ensemble (value)
        graph_predictions: dict[int, list[float]] = defaultdict(list)
        # This data structure will map the graph index (key) to a list of gradients that have been
        # calculated for the input nodes of the graph (value)
        graph_gradients: dict[int, list[float]] = defaultdict(list)
        
        for model in self.models:
            infos = model.forward_graphs(graphs)
            
            # This method will calculate the gradients of the model's output with respect to the input 
            # nodes. It returns a list of dicts with one dict per input graph that contains the key
            # 'node_gradient' with the gradient values that have the same shape as the input vector.
            grad_infos = self.gradients_graphs(model, graphs)
            
            for index, (info, grad_info) in enumerate(zip(infos, grad_infos)):
                
                # prediction: (num_outputs, )
                prediction = info['graph_output']
                # gradients: (num_nodes, num_node_features)
                gradients = grad_info['node_gradient']
                
                if self.mode == 'classification':
                    prediction = np.argmax(prediction)
                
                graph_predictions[index].append(info['graph_output'])
                
                # With this aggregation we want to somehow get rid of the variable node dimension as that would 
                # be a guaranteed source of uncertainty between different input graphs. So the aggregate_gradients
                # method will apply the aggregation method of choice to reduce this to the fixed feature dimension 
                # only.
                # gradients_agg: (num_node_features, )
                #gradients_agg: np.ndarray = self.aggregate_gradients(gradients)
                graph_gradients[index].append(gradients)
                
        results: list[dict] = []
        for index, predictions in graph_predictions.items():
            
            # Now instead of calculating the uncertainty from the predictions we actually want to calculate the 
            # uncertainty over the gradients...
            gradients = np.array(graph_gradients[index])
            gradients = gradients.reshape(gradients.shape[0], -1)
            uncertainty = np.mean(np.std(gradients, axis=0))
            
            # optionally, we apply the internal calibration mapping to the raw uncertainty value
            if calibrated and 'main' in self.calibration_models:
                uncertainty = self.calibration_models['main'].transform([uncertainty])[0]
            
            results.append({
                'avg':              np.mean(predictions),
                'std':              np.std(predictions),
                'prediction':       np.mean(predictions),
                'uncertainty':      uncertainty,
            })
            
        return results

    
class MeanVarianceUncertainty(AbstractUncertainty):
    """
    Estimates the uncertainty from the model's "variance" output directly.
    
    This class requires a model that implements the ``MeanVarianceMixin`` interface. This interface 
    ensures that the model maintains a separate output for the variance of the prediction which is 
    trained during model training time. The model must also implement a method ``predict_variance``
    that takes the graph embedding as input and returns the predicted variance.
    """
    def __init__(self, 
                 model: MeanVarianceMixin,
                 **kwargs):
        AbstractUncertainty.__init__(self, **kwargs)
        self.model = model
        
    def evaluate_graphs(self, graphs: list[dict], batch_size=128, calibrated: bool = True) -> list[dict]:
        
        loader = loader_from_graphs(graphs, batch_size=batch_size)
        
        infos = []
        for data in loader:
            info = self.model.forward(data)
            info['graph_variance'] = self.model.predict_variance(info['graph_embedding'])
            
            _infos = self.model.split_infos(info, data)
            infos += _infos
               
        results: list[dict] = []
        for info in infos:
            
            uncertainty = float(info['graph_variance'])
            # optionally, we apply the internal calibration mapping to the raw uncertainty value
            if calibrated and 'main' in self.calibration_models:
                uncertainty = self.calibration_models['main'].transform([uncertainty])[0]
            
            results.append({
                'prediction':       float(info['graph_output']),
                'uncertainty':      uncertainty,
            })
            
        return results
    
    
class EnsembleMveUncertainty(AbstractUncertainty):
    """
    Estimates the uncertainty as a composite of "mean variance estimation" and "ensemble" uncertainty.
    
    The constructor requires an ``EnsembleModel`` instance that contains models implementing the 
    ``MeanVarianceMixin`` interface. The class then evaluates the uncertainty as the sum of the ensemble 
    uncertainty (epistimic) and the mean MVE uncertainty (aleatoric).
    """
    def __init__(self,
                 ensemble: EnsembleModel[MeanVarianceMixin],
                 aggregation: Literal['mean', 'min', 'max'] = 'mean',
                 **kwargs
                 ):
        super().__init__(**kwargs)
        
        self.ensemble = ensemble
        self.models = ensemble.models
        self.aggregation = aggregation
        
    def calibrate(self, graphs: list[dict], values: list[float]):
        
        uncertainties_true = np.array([float(value) for value in values])
        
        # For an ensemble of MVE methods we need to calibtrate the MVE uncertainties of all of the ensemble 
        # members separately before we can put them on the same scale and combine them.
        model_infos = []
        for model_index, model in enumerate(self.models):
            
            infos = model.forward_graphs(graphs)
            
            # For a model that implements the MeanVarianceMixin interface, we can directly predict the variance
            # by supplying the graph embedding to the model's predict_variance method.
            uncertainties_mve = model.predict_variance(torch.tensor([info['graph_embedding'] for info in infos])).detach().numpy()
            
            self.calibration_models[model_index] = IsotonicRegression(
                y_min=np.min(uncertainties_true) - 1,
                y_max=np.max(uncertainties_true) + 1,
                out_of_bounds='clip',
                increasing='auto',
            )
            self.calibration_models[model_index].fit_transform(uncertainties_mve, uncertainties_true)
            model_infos.append(infos)
            
        predictions = np.array([[info['graph_output'] for info in infos] for infos in model_infos])
        uncertainties_ens = np.std(predictions, axis=0)
        
        self.calibration_models['ens'] = IsotonicRegression(
            y_min=np.min(uncertainties_true) - 1,
            y_max=np.max(uncertainties_true) + 1,
            out_of_bounds='clip',
            increasing='auto',
        )
        uncertainties_ens_cal = self.calibration_models['ens'].fit_transform(uncertainties_ens, uncertainties_true)
        
        return uncertainties_ens_cal, self.calibration_models
        
    def evaluate_graphs(self, 
                        graphs: list[dict], 
                        batch_size: int = 128,
                        calibrated: bool = True
                        ) -> list[dict]:
        
        loader = loader_from_graphs(graphs, batch_size=batch_size)
        
        model_infos = []
        for model in self.models:
            
            infos = []
            for data in loader:
                
                info = model.forward(data)
                
                info['graph_variance'] = model.predict_variance(info['graph_embedding'])
                
                _infos = model.split_infos(info, data)
                infos += _infos
               
            model_infos.append(infos)
               
        results: list[dict] = []
        for index in range(len(graphs)):
            predictions = [infos[index]['graph_output'] for infos in model_infos]
            variances = [infos[index]['graph_variance'] for infos in model_infos]
            
            mve_uncertainty = np.mean(variances)
            ens_uncertainty = np.std(predictions)
            
            uncertainty = mve_uncertainty + ens_uncertainty
            
            if self.calibration_models and calibrated:
                ens_uncertainty = self.calibration_models['ens'].transform([ens_uncertainty])[0]
                variances = [
                    self.calibration_models[model_index].transform([infos[index]['graph_variance']])[0]
                    for model_index, infos in enumerate(model_infos)
                ]
                mve_uncertainty = np.mean(variances)
            
            # The final uncertainty is the average of the ensemble and the mve uncertainty. This could also be 
            # the sum - in the end it doesn't really matter but using the average here doesn't violate the 
            # calibration scale...
            #print(self.aggregation)
            
            # if self.aggregation == 'mean':
            #     uncertainty = 0.5 * (mve_uncertainty + ens_uncertainty)
            # elif self.aggregation == 'min':
            #     uncertainty = min(mve_uncertainty, ens_uncertainty)
            # elif self.aggregation == 'max':
            #     uncertainty = max(mve_uncertainty, ens_uncertainty)
            
            results.append({
                'prediction':       np.mean(predictions),
                'uncertainty':      uncertainty,
                # We also return the individual uncertainties for debugging purposes
                '_ens_uncertainty': ens_uncertainty,
                '_mve_uncertainty': mve_uncertainty,
            })
            
        return results
    
    
class SwagUncertainty(AbstractUncertainty):
    """
    Estimates the uncertainty using the "stochastic weight averaging (SWAG)" method.
    
    The constructor requires a model that implements the ``SwagMixin`` interface. This interface ensures
    that the model maintains a list of "snapshots" of the model's weights that have been recorded during
    training. The model must also implement a method ``sample`` that samples a new model instance from the
    distribution of the recorded snapshots.
    
    **Stochastic Weight Averaging (SWAG)**
    
    In the SWAG method, the model maintains a list of previous weight snapshots that have been recorded 
    at different times during the later stages of model training. At the end of the training this history 
    of weights is used to estimate a weight distribution (mean & standard deviation) that can be used to 
    sample model weights from a gaussion distribution.
    To estimate uncertainty, multiple models are sampled in this fashion to perform a forward pass. The 
    prediction uncertainty is then given as the standard deviation of the predictions of the sampled models.
    """
    def __init__(self,
                 model: SwagMixin,
                 num_samples: int = 25,
                 **kwargs
                 ):
        AbstractUncertainty.__init__(self, **kwargs)
        self.model = model
        self.num_samples = num_samples
        
    def evaluate_graphs(self, 
                        graphs: list[dict], 
                        calibrated: bool = True,
                        ) -> list[dict]:
        
        graph_predictions: dict[int, list[float]] = defaultdict(list)
        for index in range(self.num_samples):
            
            # This model will overwrite the model parameters with ones that have been sampled from the 
            # distribution of parameters that has been calculated from the history of snapshots.
            model = self.model.sample()
            model.to('cpu')
            model.eval()
            
            infos = model.forward_graphs(graphs)
            for index, info in enumerate(infos):
                
                prediction = info['graph_output']
                
                if self.mode == 'classification':
                    prediction = np.argmax(prediction)
                
                graph_predictions[index].append(info['graph_output'])
                
        results: list[dict] = []
        for index, predictions in graph_predictions.items():
            
            uncertainty = np.std(predictions)
            if self.calibration_models and calibrated:
                uncertainty = self.calibration_models['main'].transform([uncertainty])[0]
            
            results.append({
                'avg':              np.mean(predictions),
                'std':              np.std(predictions),
                'prediction':       np.mean(predictions),
                'uncertainty':      np.std(predictions),
            })
            
        return results


class TrustScoreUncertainty(AbstractUncertainty):
    """
    Estimates the uncertainty using "trust scores" as described in the paper: http://arxiv.org/abs/2107.09734
    
    Trust scores are computed based on access to the training dataset as the following ratio:
    TS = (distance to nearest instance of different class) / (distance to nearest instance of same class)
    
    Higher trust scores suggest a higher confidence that the given sample is within the training distribution.
    
    :param num_neighbors: The number of neighbors in the embedding space which are considered for the 
        calculation of the trust score. Default is 1 and in this case, only the nearest instance is considered.
        For higher values, the average distance over multiple neighbors is used.
    """
    def __init__(self, 
                 model: AbstractGraphModel,
                 graphs_train: list[dict],
                 targets_train: list[float],
                 distance_metric: str = 'euclidean',
                 mode: Literal['classification', 'regression'] = 'regression',
                 num_neighbors: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.graphs_train = graphs_train
        self.targets_train = np.array(targets_train)
        self.distance_metric = distance_metric
        self.mode = mode
        self.num_neighbors = num_neighbors
        
        self._distance_metric = self.distance_metric
        if self.distance_metric == 'tanimoto':
            self._distance_metric = 'jaccard'
        
        self.embeddings_train = self._get_embeddings(graphs_train)
    
    def _get_embeddings(self, graphs: list[dict]) -> np.ndarray:
        """
        Helper method to get the embeddings of the graphs using the model.
        """
        
        # Tanimoto distance is a special case because we compute that distance based on the fingerprint 
        # representation of the molecule instead of using the graph structure.
        if self.distance_metric == 'tanimoto':
            
            smiles = [str(graph['graph_repr']) for graph in graphs]
            mols = [Chem.MolFromSmiles(s) for s in smiles]
            morgan_fps = [
                AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else np.zeros(2048)
                for mol in mols
            ]
            embeddings = np.array(morgan_fps)
            return embeddings
        
        # Otherwise we interpret the distance as being based on the graph embedding that is produced by 
        # the graph encoder model.
        else:
            infos = self.model.forward_graphs(graphs)
            embeddings = np.array([info['graph_embedding'] for info in infos])
            
        return embeddings
    
    def evaluate_graphs(self, 
                        graphs: List[dict], 
                        calibrated: bool = True,
                        ) -> List[dict]:
        """
        This method evaluates the trust scores for the given graphs.
        """
        # First of all we get the embeddings for the graphs that we want to evaluate.
        embeddings = self._get_embeddings(graphs)
        
        distances_all = pairwise_distances(
            self.embeddings_train, 
            embeddings, 
            metric=self._distance_metric
        )
        print(distances_all.shape)
        
        infos = self.model.forward_graphs(graphs)
        
        results: List[dict] = []
        for c, (graph, info, embedding) in enumerate(zip(graphs, infos, embeddings)):
            
            print(c)
            # now we have to differentiate between regression and classification tasks in 
            # the manner in which we obtain the closest instances from the train set.
            # For classification it is simple since we only need to worry about the binary 
            # decision of whether or not a sample is in the same class as the query sample 
            # or not. However, for regression we need to consider both the distance in the 
            # embedding space and the distance in the target space.
            if self.mode == 'classification':
            
                # the distances to the current embedding of all the training embeddings using 
                # the given distance metric.
                distances = distances_all[:, c]
                
                # A list of indices in the train set that have the same class as the query sample.
                indices_same = [
                    i 
                    for i, target in enumerate(self.targets_train) 
                    if np.argmax(target) == np.argmax(info['graph_output'])
                ]
                indices_other = [
                    i
                    for i, target in enumerate(self.targets_train)
                    if np.argmax(target) != np.argmax(info['graph_output'])
                ]
                
                # Selecting the self.num_neighbors with the smallest distance to the query sample
                # for the "same" and "other" cases.
                distances_same = np.array([distances[i] for i in indices_same])
                distances_other = np.array([distances[i] for i in indices_other])
                
                # We sort the distances to get the nearest neighbors.
                distances_same.sort()
                distances_other.sort()
                
                # We take the average of the self.num_neighbors nearest neighbors.
                trust_score: float = (
                    np.mean(distances_other[:self.num_neighbors]) / \
                    np.mean(distances_same[:self.num_neighbors])
                )
            
            if self.mode == 'regression':
            
                # the distances to the current embedding of all the training embeddings using 
                # the given distance metric.
                distances = distances = distances_all[:, c]
                distances_norm = distances / np.percentile(distances, 98)
                
                # We need to consider both the distance in the embedding space and the distance 
                # in the target space.
                distances_same = np.abs(self.targets_train - info['graph_output'])
                distances_same_norm = distances_same / np.percentile(distances_same, 98)
                
                # We sort the combined distances to get the nearest neighbors.
                combined_same = distances_norm + distances_same_norm
                combined_same = np.partition(combined_same, self.num_neighbors - 1)
                
                distances_other = 1 / np.abs(self.targets_train - info['graph_output'])
                distances_other_norm = distances_other / np.percentile(distances_other, 98)
                
                combined_other = distances_norm + distances_other_norm
                combined_other = np.partition(combined_other, self.num_neighbors - 1)
                
                # We take the average of the self.num_neighbors nearest neighbors.
                trust_score: float = (
                    np.mean(combined_other[:self.num_neighbors]) / \
                    np.mean(combined_same[:self.num_neighbors])
                )
            
            # The higher the trust score the more confident we are that the sample is within the 
            # training distribution. But we want to interpret this as an uncertainty value so we
            # invert the trust score
            uncertainty = 1 / trust_score
            
            # optionally, we apply the internal calibration mapping to the raw uncertainty value
            # if we have a calibration model and the calibration is enabled.
            if calibrated:
                if self.calibration_models and calibrated:
                    uncertainty = self.calibration_models['main'].transform([uncertainty])[0]
            
            # 'prediction' and 'uncertainty' are the two keys that are actually required by the 
            # downstream processing.
            prediction = info['graph_output']
            if isinstance(prediction, (np.ndarray, list)) and len(prediction) == 1:
                prediction = float(prediction[0])
            
            results.append({
                'prediction': prediction,
                'uncertainty': uncertainty,
            })
            
        return results
    

class EvidentialUncertainty(AbstractUncertainty):
    
    def __init__(self,
                 model: EvidentialMixin,
                 **kwargs
                 ):
        AbstractUncertainty.__init__(self, **kwargs)
        self.model = model
        
    def evaluate_graphs(self, graphs, calibrated = True):
        
        results: List[dict] = []
        infos = self.model.forward_graphs(graphs)
        
        for graph, info in zip(graphs, infos):
            
            uncertainty = float(info['graph_variance'][0])
            # optionally, we apply the internal calibration mapping to the raw uncertainty value
            if calibrated and 'main' in self.calibration_models:
                uncertainty = self.calibration_models['main'].transform([uncertainty])[0]
            
            results.append({
                'prediction': float(info['graph_output'][0]),
                'uncertainty': uncertainty,
            })
            
        return results