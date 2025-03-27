import os
import pathlib

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from truthful_counterfactuals.models import AbstractGraphModel, GINModel, GATModel
from truthful_counterfactuals.models import EnsembleModel
from truthful_counterfactuals.data import data_list_from_graphs
from truthful_counterfactuals.uncertainty import AbstractUncertainty
from truthful_counterfactuals.uncertainty import EnsembleMveUncertainty
from truthful_counterfactuals.uncertainty import RandomUncertainty

EPOCHS: int = 50
CALIBRATE_UNCERTAINTY: bool = False

# == ENSEMBLE PARAMETERS ==

# :param NUM_MEMBERS:
#       The number of ensemble members to train. 
NUM_MEMBERS: int = 3
# :param NUM_TEST:
#       The number of elements to be randomly sampled as the test set. The uncertainty quantification 
#       as well as the final prediction performance will be evaluated on this test set at the end.
NUM_TEST: int = 900
# :param NUM_VAL:
#       The number of elements to be randomly sampled as the validation set. This set is optionally 
#       used to calibrate the uncertainty values.
NUM_VAL: int = 10000

# == MVE PARAMETERS ==

# :param VARIANCE_UNITS:
#       The number of units in each layer of the variance predictor.
VARIANCE_UNITS = [32, 16, 1]

# == COUNTERFACTUAL PARAMETERS ==

# :param NUM_COUNTERFACTUAL_ORIGINALS:
#       The number of original elements from the test set for which counterfactuals should be generated.
NUM_COUNTERFACTUAL_ORIGINALS: int = 100
# :param NUM_COUNTERFACTUALS:
#       The number of counterfactuals to be generated from the neighborhood of each element within 
#       the test set.
NUM_COUNTERFACTUALS: int = 10

# == TRAINING PARAMETERS == 

# :param LEARNING_RATE:
#       The learning rate to be used for the model training. Determines how much the model 
#       weights are updated during each iteration of the training.
LEARNING_RATE: float = 1e-5
# :param EPOCHS:
#       The number of epochs that the model should be trained for.
EPOCHS: int = 500

__DEBUG__ = False

experiment = Experiment.extend(
    'counterfactual_truthfulness.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('train_model', default=False, replace=True)
def train_model(e: Experiment,
                index_data_map: dict[int, dict],
                train_indices: list[int],
                test_indices: list[int],
                ) -> list[tuple[pl.Trainer, AbstractGraphModel]]:

    graphs_train = [index_data_map[index]['metadata']['graph'] for index in train_indices]
    loader_train = DataLoader(data_list_from_graphs(graphs_train), batch_size=e.BATCH_SIZE, shuffle=True)
    
    e.log('starting to train model...')
        
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
                
        trainer = pl.Trainer(max_epochs=e.EPOCHS)
        trainer.fit(
            model,
            train_dataloaders=loader_train,
        )
            
        model.eval()
    
    return model


@experiment.hook('uncertainty_estimator', default=False, replace=True)
def uncertainty_estimator(e: Experiment,
                          model: AbstractGraphModel
                          ) -> AbstractUncertainty:
    
    # ~ random uncertainty
    # assigns completely random uncertainty values to the predictions
    return RandomUncertainty(model)

experiment.run_if_main()