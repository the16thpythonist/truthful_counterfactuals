import os

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from truthful_counterfactuals.models import AbstractGraphModel, GINModel, GATModel
from truthful_counterfactuals.data import data_list_from_graphs
from truthful_counterfactuals.uncertainty import AbstractUncertainty
from truthful_counterfactuals.uncertainty import SwagUncertainty
from truthful_counterfactuals.utils import EXPERIMENTS_PATH

VISUAL_GRAPH_DATASET: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp')
#VISUAL_GRAPH_DATASET: str = '/media/ssd/.visual_graph_datasets/datasets/aqsoldb'

EPOCHS: int = 50
CALIBRATE_UNCERTAINTY: bool = True
#TEST_INDICES_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp_ood_value.json')
TEST_INDICES_PATH: str = None

# == SWAG PARAMETERS ==

# :param EPOCH_START:
#       The epoch at which to start recording the model parameters for the SWAG model.
EPOCH_START: int = 30
# :param SWAG_SAMPLES:
#       The number of samples to draw from the SWAG distribution of weights for each uncertainty estimate.
#       This is the number of models that are used in the forward pass - increasing this number will linearly 
#       increase the computational cost of the inference.
SWAG_SAMPLES: int = 25

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
    
    e.log(f'using model type: {e.MODEL_TYPE}')
    if e.MODEL_TYPE.lower() == 'gin':
        model = GINModel(
            node_dim=e['node_dim'],
            edge_dim=e['edge_dim'],
            encoder_units=e.ENCODER_UNITS,
            predictor_units=e.PREDICTOR_UNITS,
            mode=e.DATASET_TYPE,
            learning_rate=e.LEARNING_RATE,
            epoch_start=int(0.8 * e.EPOCHS),
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
            epoch_start=e.EPOCH_START,
        )
       
    # ~ mve training
    # When training a model for mean-variance estimation, we need to follow a slightly different 
    # protocol. It is generally recommended to use a warm-up period where the model is only trained 
    # with the prediction loss and only afterwards to add the variance loss to the training. 
    e.log('starting warm-up training...')
    trainer = pl.Trainer(max_epochs=e.EPOCHS)
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
    
    return SwagUncertainty(
        model=model, 
        mode=e.DATASET_TYPE,
        num_samples=e.SWAG_SAMPLES
    )


experiment.run_if_main()