import os

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from truthful_counterfactuals.models import EnsembleModel
from truthful_counterfactuals.models import AbstractGraphModel, GINModel, GATModel
from truthful_counterfactuals.data import data_list_from_graphs
from truthful_counterfactuals.uncertainty import AbstractUncertainty
from truthful_counterfactuals.uncertainty import EnsembleGradientUncertainty
from truthful_counterfactuals.utils import EXPERIMENTS_PATH

VISUAL_GRAPH_DATASET: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp')
#VISUAL_GRAPH_DATASET: str = '/media/ssd/.visual_graph_datasets/datasets/aqsoldb'
#TEST_INDICES_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp_ood_value.json')
TEST_INDICES_PATH: str = None
NUM_TEST: int = 3000

EPOCHS: int = 50
CALIBRATE_UNCERTAINTY: bool = True

# == ENSEMBLE PARAMETERS ==

# :param NUM_MEMBERS:
#       The number of ensemble members to train. 
NUM_MEMBERS: int = 3

__DEBUG__ = False

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

    graphs_train = [index_data_map[index]['metadata']['graph'] for index in train_indices]
    loader_train = DataLoader(data_list_from_graphs(graphs_train), batch_size=e.BATCH_SIZE, shuffle=True)
    
    e.log('starting to train ensemble...')
    models: list[AbstractGraphModel] = []
        
    for index in range(e.NUM_MEMBERS):
        e.log(f'starting training model {index}...')
        
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
        models.append(model)
    
    e.log('assembling the model into an ensemble...')
    ensemble = EnsembleModel(models)
    
    return ensemble


@experiment.hook('uncertainty_estimator', default=False, replace=True)
def uncertainty_estimator(e: Experiment,
                          model: AbstractGraphModel
                          ) -> AbstractUncertainty:
    
    # ~ mixed uncertainty: MVE + Ensemble
    # This uncertainty class computes the final uncertainty value as the sum of the ensemble 
    # uncertainty and the MVE uncertainty. The ensemble uncertainty is computed as the standard
    # deviation of the ensemble predictions, while the MVE uncertainty is computed via the variance 
    # estimation network that is part of the MVE models.
    return EnsembleGradientUncertainty(
        ensemble=model,
        aggregation_method='mean',
    )


experiment.run_if_main()