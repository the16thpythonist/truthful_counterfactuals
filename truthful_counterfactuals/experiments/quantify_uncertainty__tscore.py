"""
This experiment implements the uncertainty quantification with the help of "trust scores" as 
it is described in the paper: http://arxiv.org/abs/2107.09734 

Specifically, the trust scores are computed based on access to the training dataset as the 
following ratio:

TS = (distance to nearest instance of different class) / (distance to nearest instance of same class)

higher trust scores suggest a higher confidence that the given sample is within the training distribution.
"""
import os
from typing import List, Dict

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from truthful_counterfactuals.uncertainty import TrustScoreUncertainty
from truthful_counterfactuals.models import AbstractGraphModel, GINModel, GATModel
from truthful_counterfactuals.data import data_list_from_graphs
from truthful_counterfactuals.uncertainty import AbstractUncertainty
from truthful_counterfactuals.utils import EXPERIMENTS_PATH

VISUAL_GRAPH_DATASET: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp')
TEST_INDICES_PATH: str = None
NUM_TEST: int = 300
NUM_VAL: int = 200
MODEL_TYPE: str = 'gin'

EPOCHS: int = 50
CALIBRATE_UNCERTAINTY: bool = False

# :param DISTANCE_METRIC:
#       The distance metric to use when computing the trust scores. This can be any of the distance metrics
#       supported by the `sklearn.metrics.pairwise_distances` function or the special 'tanimoto' metric which
#       computes the tanimoto distance between the fingerprints of the molecules.
DISTANCE_METRIC: str = 'tanimoto'

__DEBUG__ = True

experiment = Experiment.extend(
    'quantify_uncertainty.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('uncertainty_estimator', default=False, replace=True)
def uncertainty_estimator(e: Experiment,
                          model: AbstractGraphModel,
                          index_data_map: Dict[int, dict],
                          train_indices: List[int],
                          **kwargs,
                          ) -> AbstractUncertainty:
    """
    This hook constructs the concrete AbstractUncertainty instance that can be used 
    to quantify the uncertainty of the model predictions. This hook should return an 
    instance which implements the AbstractUncertainty interface.
    
    ---
    
    This implementation returns a TrustScoreUncertainty instance with the given model.
    """
    graphs_train = [index_data_map[index]['metadata']['graph'] for index in train_indices]
    targets_train = [index_data_map[index]['metadata']['target'] for index in train_indices]
    estimator = TrustScoreUncertainty(
        model=model,
        graphs_train=graphs_train,
        targets_train=targets_train,
        distance_metric=e.DISTANCE_METRIC,
    )
    return estimator


experiment.run_if_main()
