import os
from typing import List, Dict

import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from truthful_counterfactuals.models import AbstractGraphModel, GINModel, GATModel
from truthful_counterfactuals.data import data_list_from_graphs
from truthful_counterfactuals.uncertainty import AbstractUncertainty
from truthful_counterfactuals.uncertainty import EvidentialUncertainty
from truthful_counterfactuals.utils import EXPERIMENTS_PATH

VISUAL_GRAPH_DATASET: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp')
#VISUAL_GRAPH_DATASET: str = '/media/ssd/.visual_graph_datasets/datasets/aqsoldb'

EPOCHS: int = 50
CALIBRATE_UNCERTAINTY: bool = True
TEST_INDICES_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp_ood_value.json')

__DEBUG__ = True

experiment = Experiment.extend(
    'quantify_uncertainty.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('train_model', default=False, replace=True)
def train_model(e: Experiment,
                index_data_map: dict[int, dict],
                train_indices: list[int],
                test_indices: list[int],
                **kwargs,
                ) -> AbstractGraphModel:
    """
    This hook is meant to train a new model given the information about the dataset contained in the 
    ``index_data_map`` and the information about the train/test split in ``train_indices`` and 
    ``test_indices``. The hook should return a fully trained model which implements the 
    AbstractGraphModel interface.
    
    ---
    
    This implementation trains a new model using the MVE loss until the resulting model is valid 
    which means that there is no infinity or nan in the model's predictions.
    """
    
    graphs_train = [index_data_map[index]['metadata']['graph'] for index in train_indices]
    
    # We dynamically choose the specific hook to invoke based on the string model type specified 
    # as a parameter. This allows us to easily switch between different model types without having
    # to change the code in multiple places.
    model_type = e.MODEL_TYPE.lower()
    e.log(f'training model of type "{model_type}" for {e.EPOCHS} epochs...')
    
    is_valid = False
    while not is_valid:
        
        model = e.apply_hook(
            f'train_model__{model_type}',
            index_data_map=index_data_map,
            train_indices=train_indices,
            test_indices=test_indices,    
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
    
    return model


@experiment.hook('train_model__gat', default=False, replace=True)
def train_model__gat(e: Experiment,
                     index_data_map: dict,
                     train_indices: List[int],
                     test_indices: List[int],
                     **kwargs,
                     ) -> AbstractGraphModel:
    
    graphs_train = [index_data_map[index]['metadata']['graph'] for index in train_indices]
    loader_train = DataLoader(data_list_from_graphs(graphs_train), batch_size=e.BATCH_SIZE, shuffle=True)
        
    model = GATModel(
        node_dim=e['node_dim'],
        edge_dim=e['edge_dim'],
        num_heads=3,
        encoder_units=e.ENCODER_UNITS,
        predictor_units=e.PREDICTOR_UNITS,
        mode=e.DATASET_TYPE,
        learning_rate=e.LEARNING_RATE,
        use_evidential_regression=True,
    )
        
    e.log(f'starting training for {e.EPOCHS} epochs...')
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS
    )
    trainer.fit(
        model,
        train_dataloaders=loader_train,
    )
        
    e.log('model training done...')
    model.eval()
    
    return model


@experiment.hook('train_model__gin', default=False, replace=True)
def train_model__gin(e: Experiment,
                     index_data_map: dict,
                     train_indices: List[int],
                     test_indices: List[int],
                     **kwargs,
                     ) -> AbstractGraphModel:
    
    graphs_train = [index_data_map[index]['metadata']['graph'] for index in train_indices]
    loader_train = DataLoader(data_list_from_graphs(graphs_train), batch_size=e.BATCH_SIZE, shuffle=True)
        
    model = GINModel(
        node_dim=e['node_dim'],
        edge_dim=e['edge_dim'],
        encoder_units=e.ENCODER_UNITS,
        predictor_units=e.PREDICTOR_UNITS,
        mode=e.DATASET_TYPE,
        learning_rate=e.LEARNING_RATE,
    )
        
    e.log(f'starting training for {e.EPOCHS} epochs...')
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS
    )
    trainer.fit(
        model,
        train_dataloaders=loader_train,
    )
        
    e.log('model training done...')
    model.eval()
    
    return model


@experiment.hook('uncertainty_estimator', default=False, replace=True)
def uncertainty_estimator(e: Experiment,
                          model: AbstractGraphModel,
                          **kwargs,
                          ) -> AbstractUncertainty:
    
    return EvidentialUncertainty(model)


experiment.run_if_main()