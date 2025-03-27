import os
import random
from typing import List, Dict

import torch
import pytorch_lightning as pl
from rich import print as pprint
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from truthful_counterfactuals.models import EnsembleModel
from truthful_counterfactuals.models import AbstractGraphModel, GINModel, GATModel
from truthful_counterfactuals.data import data_list_from_graphs
from truthful_counterfactuals.uncertainty import AbstractUncertainty
from truthful_counterfactuals.uncertainty import EnsembleMveUncertainty
from truthful_counterfactuals.utils import EXPERIMENTS_PATH

VISUAL_GRAPH_DATASET: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp')
#VISUAL_GRAPH_DATASET: str = '/media/ssd/.visual_graph_datasets/datasets/aqsoldb'
#TEST_INDICES_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp_ood_value.json')
#TEST_INDICES_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp_ood_struct.json')
TEST_INDICES_PATH: str = None

EPOCHS: int = 50
CALIBRATE_UNCERTAINTY: bool = True

# == MVE PARAMETERS ==

# :param VARIANCE_UNITS:
#       The number of units in each layer of the variance predictor.
VARIANCE_UNITS = [32, 16, 1]

# == ENSEMBLE PARAMETERS ==

# :param USE_BAGGING:
#       Whether or not to use bootstrap aggregation. If this is set to True, the training dataset 
#       will be sampled with replacement to create multiple training sets for each model in the
#       ensemble.
USE_BAGGING: bool = True
# :param NUM_MEMBERS:
#       The number of ensemble members to train. 
NUM_MEMBERS: int = 5

# :param UNCERTAINTY_AGGREGATION:
#       The method to aggregate the uncertainty values of the ensemble method and the MVE
#       method. The possible values are 'mean', 'max', 'min'. 
UNCERTAINTY_AGGREGATION: str = 'mean'


__DEBUG__ = True

experiment = Experiment.extend(
    'quantify_uncertainty__mve.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

#@experiment.hook('train_model', default=False, replace=True)
def train_model(e: Experiment,
                index_data_map: dict[int, dict],
                train_indices: list[int],
                test_indices: list[int],
                ) -> list[tuple[pl.Trainer, AbstractGraphModel]]:

    graphs_train = [index_data_map[index]['metadata']['graph'] for index in train_indices]
    loader_train = DataLoader(data_list_from_graphs(graphs_train), batch_size=e.BATCH_SIZE, shuffle=True)
    
    e.log('starting to train ensemble...')
    models: list[AbstractGraphModel] = []
        
    for index in range(e.NUM_MEMBERS):
        e.log(f'starting training model ({index}/{e.NUM_MEMBERS})...')
        
        is_valid = False
        while not is_valid:
        
            e.log(f'using model type: {e.MODEL_TYPE}')
            if e.MODEL_TYPE.lower() == 'gin':
                model = GINModel(
                    node_dim=e['node_dim'],
                    edge_dim=e['edge_dim'],
                    encoder_units=e.ENCODER_UNITS,
                    predictor_units=e.PREDICTOR_UNITS,
                    mode=e.DATASET_TYPE,
                    learning_rate=e.LEARNING_RATE,
                )
                
            elif e.MODEL_TYPE.lower() == 'gat':
                model = GATModel(
                    node_dim=e['node_dim'],
                    edge_dim=e['edge_dim'],
                    num_heads=3,
                    encoder_units=e.ENCODER_UNITS,
                    predictor_units=e.PREDICTOR_UNITS,
                    mode=e.DATASET_TYPE,
                    learning_rate=e.LEARNING_RATE,
                )
                
            # ~ mve training
            # When training a model for mean-variance estimation, we need to follow a slightly different 
            # protocol. It is generally recommended to use a warm-up period where the model is only trained 
            # with the prediction loss and only afterwards to add the variance loss to the training. 
        
            e.log('starting warm-up training...')
            trainer = pl.Trainer(max_epochs=e.EPOCHS // 2)
            trainer.fit(
                model,
                train_dataloaders=loader_train,
            )
            
            e.log('switching to mean-variance training...')
            model.enable_variance_training()
            trainer = pl.Trainer(
                max_epochs=e.EPOCHS // 2, 
                # Some degree of gradient clipping is recommended for MVE training to increase the 
                # stability / prevent the exploding gradients problem
                gradient_clip_val=50.0, 
                gradient_clip_algorithm='value'
            )
            trainer.fit(
                model,
                train_dataloaders=loader_train,
            )

            # At the end of the training here we need to check if the resulting MVE model 
            # is actually valid (i.e. the training did not diverge). We do this by checking
            # if the output of the model is not NaN. If it is, we simply try again to train
            # a new model until one is valid.
            data = next(iter(DataLoader(data_list_from_graphs(graphs_train[0:]))))
            info = model.forward(data)
            var = model.predict_variance(info['graph_embedding'])
            is_valid = not torch.isnan(info['graph_output']) and not torch.isnan(var)
            e.log(f' * valid: {is_valid}')
        
        model.eval()
        models.append(model)
    
    e.log('assembling the model into an ensemble...')
    ensemble = EnsembleModel(models)
    
    return ensemble


@experiment.hook('train_model', default=False, replace=True)
def train_model(e: Experiment,
                index_data_map: dict[int, dict],
                train_indices: list[int],
                test_indices: list[int],
                ) -> list[tuple[pl.Trainer, AbstractGraphModel]]:
    
    graphs_train = [index_data_map[index]['metadata']['graph'] for index in train_indices]
    
    e.log('starting to train ensemble...')
    models: list[AbstractGraphModel] = []
        
    for index in range(e.NUM_MEMBERS):
        e.log(f'starting training model {index}...')
        
        if e.USE_BAGGING:
            e.log('using bagging, sampling train indices with replacement...')
            _train_indices = random.choices(train_indices, k=len(train_indices))
        else:
            _train_indices = train_indices
        
        # With MVE training there is the real possibility that the training loss diverges and 
        # the result is a non-usable model. In this case we simply try again to train a new model 
        # until one is valid.
        is_valid = False
        while not is_valid:
            
            # We dynamically choose the specific hook to invoke based on the string model type specified 
            # as a parameter. This allows us to easily switch between different model types without having
            # to change the code in multiple places.
            model_type = e.MODEL_TYPE.lower()
            e.log(f'training model of type "{model_type}"...')
            model = e.apply_hook(
                f'train_model__{model_type}',
                index_data_map=index_data_map,
                train_indices=_train_indices,
                test_indices=test_indices,    
            )
            
             # At the end of the training there
            data = next(iter(DataLoader(data_list_from_graphs(graphs_train[0:]))))
            info = model.forward(data)
            var = model.predict_variance(info['graph_embedding'])
            is_valid = not torch.isnan(info['graph_output']) and not torch.isnan(var)
            e.log(f' * valid: {is_valid}')
            
        models.append(model)
    
    e.log('assembling the model into an ensemble...')
    ensemble = EnsembleModel(models)
    
    return ensemble


@experiment.hook('uncertainty_estimator', default=False, replace=True)
def uncertainty_estimator(e: Experiment,
                          model: AbstractGraphModel,
                          **kwargs,
                          ) -> AbstractUncertainty:
    
    # ~ mixed uncertainty: MVE + Ensemble
    # This uncertainty class computes the final uncertainty value as the sum of the ensemble 
    # uncertainty and the MVE uncertainty. The ensemble uncertainty is computed as the standard
    # deviation of the ensemble predictions, while the MVE uncertainty is computed via the variance 
    # estimation network that is part of the MVE models.
    return EnsembleMveUncertainty(model, aggregation=e.UNCERTAINTY_AGGREGATION)


experiment.run_if_main()