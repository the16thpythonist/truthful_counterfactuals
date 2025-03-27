"""
Derivative of the main ``quantify_uncertainty`` experiment which uses repulsive ensemble as 
the uncertainty quantification method.
"""
import os
from typing import List, Dict

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from truthful_counterfactuals.models import AbstractGraphModel, GINModel, GATModel
from truthful_counterfactuals.models import RepulsiveEnsembleModel
from truthful_counterfactuals.data import data_list_from_graphs
from truthful_counterfactuals.uncertainty import AbstractUncertainty
from truthful_counterfactuals.uncertainty import SwagUncertainty
from truthful_counterfactuals.uncertainty import EnsembleUncertainty
from truthful_counterfactuals.utils import EXPERIMENTS_PATH

VISUAL_GRAPH_DATASET: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp')
#VISUAL_GRAPH_DATASET: str = '/media/ssd/.visual_graph_datasets/datasets/aqsoldb'

CALIBRATE_UNCERTAINTY: bool = True
TEST_INDICES_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp_ood_value.json')
#TEST_INDICES_PATH: str = None

# == TRAINING PARAMETERS == 

# :param LEARNING_RATE:
#       The learning rate to be used for the model training. Determines how much the model 
#       weights are updated during each iteration of the training.
LEARNING_RATE: float = 1e-4
# :param EPOCHS:
#       The number of epochs that the model should be trained for.
EPOCHS: int = 50

# == MODEL PARAMETERS ==

# :param MODEL_TYPE:
#       The type of the model to be used for the experiment. This can be either 'GIN' or 'GAT'.
#       The model type determines the architecture of the model that is used for the experiment.
MODEL_TYPE: str = 'gin'
# :param ENCODER_UNITS:
#       The number of units to be used in the encoder part of the model. This essentially determines
#       the number of neurons in each layer of the message passing encoder subnetwork.
ENCODER_UNITS = [128, 128, 128]
# :param PREDICTOR_UNITS:
#       The number of units to be used in the predictor part of the model. This essentially determines
#       the number of neurons in each layer of the final prediction subnetwork.
PREDICTOR_UNITS = [256, 128, 64, 1]

# == ENSEMBLE PARAMETERS ==

# :param REPULSIVE_FACTOR:
#       The factor by which to repulse the individual models in the ensemble. This value 
#       will be used as the weight of the repulsion term during the model training.
REPULSIVE_FACTOR: float = 1e-3
# :param NUM_MEMBERS:
#       The number of ensemble members to train. 
NUM_MEMBERS: int = 5


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
                 ) -> AbstractGraphModel:

    graphs_train = [index_data_map[index]['metadata']['graph'] for index in train_indices]
    loader_train = DataLoader(data_list_from_graphs(graphs_train), batch_size=e.BATCH_SIZE, shuffle=True)
    
    # We'll store the individual initialized models in this list.
    models: List[AbstractGraphModel] = []
    e.log(f'using model type: {e.MODEL_TYPE}')
    
    for i in range(e.NUM_MEMBERS):
        
        if e.MODEL_TYPE.lower() == 'gin':
            model = GINModel(
                node_dim=e['node_dim'],
                edge_dim=e['edge_dim'],
                encoder_units=e.ENCODER_UNITS,
                predictor_units=e.PREDICTOR_UNITS,
                mode=e.DATASET_TYPE,
                learning_rate=e.LEARNING_RATE,
                hidden_units=512,
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
            
        models.append(model)
       
    e.log('constructing the ensemble model...')
    model = RepulsiveEnsembleModel(
        models=models,
        repulsive_factor=e.REPULSIVE_FACTOR,
    )
    
    class TrackingCallback(pl.Callback):
        
        def on_train_epoch_end(self, trainer, pl_module):
            super().on_train_epoch_end(trainer, pl_module)
            
            # Iterate through all logged values
            for key, value in trainer.callback_metrics.items():
                if value is not None and 'epoch' in key:
                    # Track each value using the experiment's track method
                    e.track(key, value.item())
       
    e.log(f'starting training with {e.EPOCHS} epochs...')
    trainer = pl.Trainer(max_epochs=e.EPOCHS, callbacks=[TrackingCallback()])
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
    
    return EnsembleUncertainty(
        model, 
        mode=e.DATASET_TYPE,
    )


experiment.run_if_main()