import os

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from truthful_counterfactuals.utils import EXPERIMENTS_PATH


# == SOURCE PARAMETERS ==

# :param VISUAL_GRAPH_DATASET:
#       The path to the visual graph dataset folder that should be used as the basis of the 
#       experiment.
VISUAL_GRAPH_DATASET: str = '/media/ssd/.visual_graph_datasets/datasets/qm9'
# :param TEST_INDICES_PATH:
#       The path to a file containing the indices of the test set. If this is set, the test set
#       will be loaded from this file instead of being randomly sampled. The file should be a 
#       single JSON list of integers.
TEST_INDICES_PATH: str = None
# :param NUM_TEST:
#       The number of elements to be randomly sampled as the test set. The uncertainty quantification 
#       as well as the final prediction performance will be evaluated on this test set at the end.
NUM_TEST: int = 1000
# :param NUM_VAL:
#       The number of elements to be randomly sampled as the validation set. This set is optionally 
#       used to calibrate the uncertainty values.
NUM_VAL: int = 1000
# :param DATASET_TYPE:
#       The type of the dataset. This can be either 'regression' or 'classification'.
#       Currently only regression supported!
DATASET_TYPE: str = 'regression'

# == MODEL PARAMETERS ==

# :param MODEL_TYPE:
#       The type of the model to be used for the experiment. This can be either 'GIN' or 'GAT'.
#       The model type determines the architecture of the model that is used for the experiment.
MODEL_TYPE: str = 'GAT'
# :param ENCODER_UNITS:
#       The number of units to be used in the encoder part of the model. This essentially determines
#       the number of neurons in each layer of the message passing encoder subnetwork.
ENCODER_UNITS = [64, 64, 64]
# :param PREDICTOR_UNITS:
#       The number of units to be used in the predictor part of the model. This essentially determines
#       the number of neurons in each layer of the final prediction subnetwork.
PREDICTOR_UNITS = [32, 16, 1]

# == TRAINING PARAMETERS == 

# :param LEARNING_RATE:
#       The learning rate to be used for the model training. Determines how much the model 
#       weights are updated during each iteration of the training.
LEARNING_RATE: float = 1e-5
# :param EPOCHS:
#       The number of epochs that the model should be trained for.
EPOCHS: int = 50
# :param BATCH_SIZE:
#       The batch size to be used for the model training. Determines how many samples are
#       processed in each iteration of the training.
BATCH_SIZE: int = 128
# :param CALIBRATE_UNCERTAINTY:
#       Whether the uncertainty values should be calibrated using the validation set.
#       A calibration would mean that the predicted uncertainties and the true uncertainties 
#       (prediction error) are compared and a separate calibration model is fitted to 
#       adjust the predicted uncertainties to better match the true uncertainties.
CALIBRATE_UNCERTAINTY: bool = True

# == VISUALIZATION PARAMETERS ==

# :param FIG_SIZE:
#       The size of the figures that are generated during the experiment.
#       This value will be used both the width and the height of the plots.
#       This essentially determines the aspect ratio of how large the various text 
#       elements will appear within the plot.
FIG_SIZE: int = 5

# == DATASET PARAMETERS ==

# :param TARGET:
#       The target property that should be predicted by the model. This can either be 
#       'energy' or 'dipole'
TARGET: str = 'energy'


experiment = Experiment.extend(
    'quantify_uncertainty__ens_mve.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('after_dataset', replace=False, default=False)
def after_dataset(e: Experiment,
                  index_data_map: dict[int, dict],
                  processing: any,
                  ) -> None:
    
    for index, data in index_data_map.items():
        
        if e.TARGET == 'energy':
            # This label (index 10) is the internal energy at 0K U_0
            data['metadata']['graph']['graph_labels'] = data['metadata']['graph']['graph_labels'][10:11]
        elif e.TARGET == 'dipole':
            # This label (index 3) is the dipole moment mu
            data['metadata']['graph']['graph_labels'] = data['metadata']['graph']['graph_labels'][3:4]


experiment.run_if_main()