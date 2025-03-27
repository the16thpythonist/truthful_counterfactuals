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
from truthful_counterfactuals.uncertainty import EnsembleMveUncertainty

EPOCHS: int = 50
CALIBRATE_UNCERTAINTY: bool = True

# == ENSEMBLE PARAMETERS ==

# :param NUM_MEMBERS:
#       The number of ensemble members to train. 
NUM_MEMBERS: int = 5
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
# :param USE_BAGGING:
#       Whether or not to use bootstrap aggregation. If this is set to True, the training dataset
#       will be sampled with replacement to create multiple training sets for each model in the
#       ensemble.
USE_BAGGING: bool = True

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
    'counterfactual_truthfulness__mve.py',
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
    
    e.log('starting to train ensemble...')
    models: list[AbstractGraphModel] = []
        
    for index in range(e.NUM_MEMBERS):
        e.log(f'starting training model {index}...')
        
        if e.USE_BAGGING:
            e.log('using bagging, sampling train indices with replacement...')
            _train_indices = random.choices(train_indices, k=int(0.9 * len(train_indices)))
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



@experiment.hook('uncertainty_estimator', default=False, replace=True)
def uncertainty_estimator(e: Experiment,
                          model: AbstractGraphModel
                          ) -> AbstractUncertainty:
    
    # ~ mixed uncertainty: MVE + Ensemble
    # This uncertainty class computes the final uncertainty value as the sum of the ensemble 
    # uncertainty and the MVE uncertainty. The ensemble uncertainty is computed as the standard
    # deviation of the ensemble predictions, while the MVE uncertainty is computed via the variance 
    # estimation network that is part of the MVE models.
    return EnsembleMveUncertainty(model)

experiment.run_if_main()
