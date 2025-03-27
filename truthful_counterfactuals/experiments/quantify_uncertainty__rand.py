import os

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from truthful_counterfactuals.models import AbstractGraphModel, GINModel, GATModel
from truthful_counterfactuals.data import data_list_from_graphs
from truthful_counterfactuals.uncertainty import RandomUncertainty
from truthful_counterfactuals.utils import EXPERIMENTS_PATH

VISUAL_GRAPH_DATASET: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp')
#VISUAL_GRAPH_DATASET: str = '/media/ssd/.visual_graph_datasets/datasets/aqsoldb'

EPOCHS: int = 50
CALIBRATE_UNCERTAINTY: bool = False


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
    
    e.log('starting warm-up training...')
        
    # model = GINModel(
    #     node_dim=e['node_dim'],
    #     edge_dim=e['edge_dim'],
    #     encoder_units=e.ENCODER_UNITS,
    #     predictor_units=e.PREDICTOR_UNITS,
    #     variance_units=e.VARIANCE_UNITS,
    #     mode=e.DATASET_TYPE,
    #     learning_rate=e.LEARNING_RATE,
    # )
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
        
    e.log('model training done...')
    model.eval()
    
    return model


@experiment.hook('quantify_uncertainty', default=False, replace=True)
def quantify_uncertainty(e: Experiment,
                         model: AbstractGraphModel,
                         index_data_map: dict[int, dict],
                         indices: list[int],
                         ) -> list[dict]:
    
    graphs = [index_data_map[index]['metadata']['graph'] for index in indices]
    
    uncertainty_estimator = RandomUncertainty(model, mode=e.DATASET_TYPE)
    results: list[dict] = uncertainty_estimator.evaluate_graphs(graphs)

    return results


experiment.run_if_main()