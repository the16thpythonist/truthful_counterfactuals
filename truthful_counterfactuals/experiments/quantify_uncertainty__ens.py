import os
import random

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from truthful_counterfactuals.models import EnsembleModel
from truthful_counterfactuals.models import AbstractGraphModel, GINModel, GATModel
from truthful_counterfactuals.data import data_list_from_graphs
from truthful_counterfactuals.uncertainty import AbstractUncertainty
from truthful_counterfactuals.uncertainty import EnsembleUncertainty
from truthful_counterfactuals.uncertainty import EnsembleGradientUncertainty
from truthful_counterfactuals.utils import EXPERIMENTS_PATH

VISUAL_GRAPH_DATASET: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp')
#VISUAL_GRAPH_DATASET: str = '/media/ssd/.visual_graph_datasets/datasets/aqsoldb'
TEST_INDICES_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp_ood_struct.json')
TEST_INDICES_PATH = None

EPOCHS: int = 50
CALIBRATE_UNCERTAINTY: bool = True

# == ENSEMBLE PARAMETERS ==

# :param USE_BAGGING:
#       Whether or not to use bootstrap aggregation. If this is set to True, the training dataset 
#       will be sampled with replacement to create multiple training sets for each model in the
#       ensemble.
USE_BAGGING: bool = True
# :param NUM_MEMBERS:
#       The number of ensemble members to train. 
NUM_MEMBERS: int = 3

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
                ) -> list[tuple[pl.Trainer, AbstractGraphModel]]:
    
    e.log('starting to train ensemble...')
    models: list[AbstractGraphModel] = []
        
    for index in range(e.NUM_MEMBERS):
        e.log(f'starting training model {index}...')
        
        if e.USE_BAGGING:
            e.log('using bagging, sampling train indices with replacement...')
            _train_indices = random.choices(train_indices, k=len(train_indices))
        else:
            _train_indices = train_indices
        
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
        models.append(model)
    
    e.log('assembling the model into an ensemble...')
    ensemble = EnsembleModel(models)
    
    return ensemble


@experiment.hook('uncertainty_estimator', default=False, replace=True)
def uncertainty_estimator(e: Experiment,
                          model: AbstractGraphModel,
                          **kwargs,
                          ) -> AbstractUncertainty:
    
    # ~ ensemble uncertainty
    # This class is responsible for providing the uncertainty estimates for the ensemble of models 
    # by using the standard deviation of the ensemble predictions as the uncertainty estimate.
    return EnsembleUncertainty(model)


experiment.run_if_main()