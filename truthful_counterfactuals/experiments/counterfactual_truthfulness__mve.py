import os
import pathlib
import random
from typing import List

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
from truthful_counterfactuals.uncertainty import MeanVarianceUncertainty

EPOCHS: int = 50
CALIBRATE_UNCERTAINTY: bool = True

# == ENSEMBLE PARAMETERS ==

# :param NUM_TEST:
#       The number of elements to be randomly sampled as the test set. The uncertainty quantification 
#       as well as the final prediction performance will be evaluated on this test set at the end.
NUM_TEST: int = 0.1
# :param NUM_VAL:
#       The number of elements to be randomly sampled as the validation set. This set is optionally 
#       used to calibrate the uncertainty values.
NUM_VAL: int = 0.2
# :param NUM_TRAIN:
#       The number of elements to be used as the training set. The remaining elements will be used
#       as the test set.
NUM_TRAIN: int = None
# :param MODEL_TYPE:
#       The type of the model to be used for the experiment. This can be either 'gin' or 'gat'.
#       The model type determines the architecture of the model that is used for the experiment.
MODEL_TYPE: str = 'gat'

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
EPOCHS: int = 50

__DEBUG__ = True

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
    e.log(f'training model of type "{model_type}"...')
    
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
        variance_units=e.VARIANCE_UNITS,
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
        
    e.log('model training done...')
    model.eval()
    
    # very important: At the end of the training we need to put the model into evaluation mode!
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
        variance_units=e.VARIANCE_UNITS,
        epoch_start=int(0.8 * e.EPOCHS),
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
        
    e.log('model training done...')
    model.eval()
    
    # very important: At the end of the training we need to put the model into evaluation mode!
    model.eval()
    
    return model


@experiment.hook('uncertainty_estimator', default=False, replace=True)
def uncertainty_estimator(e: Experiment,
                          model: AbstractGraphModel
                          ) -> AbstractUncertainty:
    
    # ~ mixed uncertainty: MVE + Ensemble
    # This uncertainty class computes the final uncertainty value as the sum of the ensemble 
    # uncertainty and the MVE uncertainty. The ensemble uncertainty is computed as the standard
    # deviation of the ensemble predictions, while the MVE uncertainty is computed via the variance 
    # estimation network that is part of the MVE models.
    return MeanVarianceUncertainty(model)

experiment.run_if_main()
