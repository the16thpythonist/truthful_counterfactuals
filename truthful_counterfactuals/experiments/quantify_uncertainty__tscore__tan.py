"""
This sub-experiment extends the uncertainty quantification experiment using trust scores
with the "tanimoto" distance metric. The tanimoto distance is used to compute the trust scores
based on the fingerprints of the molecules.

The trust scores are computed as the ratio:
TS = (distance to nearest instance of different class) / (distance to nearest instance of same class)

Higher trust scores suggest a higher confidence that the given sample is within the training distribution.
"""
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from truthful_counterfactuals.utils import EXPERIMENTS_PATH

DISTANCE_METRIC: str = 'tanimoto'

experiment = Experiment.extend(
    'quantify_uncertainty__tscore.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

experiment.run_if_main()
