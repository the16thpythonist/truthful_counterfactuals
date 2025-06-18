# Computational Experiments

All modules in this folder are scripts which implement their own computational experiments, which can be
executed to make some calculations and subsequently produce some results.

All the experiments in this folder are implemented with a special library called ``pycomex``. The core
feature of this library is the automatic management of archive folders for each of the experiments. When
an experiment is executed, this library will automatically create a new archive folder within the ``results``
folder. Inside these archive folders all the results and artifacts created by the experiments are
persistently stored. Pycomex offers various other advanced features such as decoupled analysis execution
and experiment inheritance which may or may not be important to fully understand the implementations of
all the experiments in this folder. For more information visit: https://github.com/the16hpythonist/pycomex

## List of Experiment Modules

The following list gives a brief overview of the experiments and their purpose:

- ``quantify_uncertainty.py``: Base experiment for the uncertainty quantification for placeholder models 
  and uncertainty quantification methods.
- ``quantify_uncertainty__ens.py``: Experiment which uses model ensembles for the uncertainty quantification.


## List of Experiment Sweeps

The following provides a list of *meta experiements*

- **Experiment 1**. Investigates the difference of combinations of uncertainty quantification methods and GNN model architectures for 5 independent seeds on the ClogP dataset.
  - ``ex_1_a`` Initial resutls for GIN and GAT architectures with MVE, ENS, TS, SWAG.
  - ``ex_1_i`` Includes MVE & ENS+MVE with GIN and GAT architectures. This uses the "fixed" version of MVE and the inclusion of the combined method as prompted by the reviewers.
  - ``ex_1_j`` Includes the results for the newly added GCN architecture for all of the methods: MVE, ENS, TS, SWAG, ENS+MVE. This was added in response to the reviews.
- **Experiment 2**. Investigates the effect of different IID/OOD scenarios on a single architecture and 
  with a small selection of methods.
  - ``ex_2_i`` A rerun of the experiment with MVE, ENS and MVE+ENS on the GAT architecture but now using the 
  "fixed" version of MVE.
- **Experiment 3**. Runs a single combination of model architecture and uncertainty quantification method on 
  various different molecular property prediction datasets for 5 independent seeds.
  - ``ex_3_a`` Initial results for the GAT architecture with ENS+MVE on all datasets.
  - ``ex_3_i`` Updated results using the "fixed" version of MVE using the GAT architecture with ENS+MVE on all datasets.
- **Experiment 4**. Experiment on the counterfactual truthfulness. This experiment uses the ClogP dataset 
  because that property can be deterministically calculated by RDKit. It constructs an initial set of counterfactual explanations for some of the elements in the test set and then predicts the uncertainty for those. A 
  uncertainty threshold is used to filter out the counterfactuals which are not truthful. The experiment 
  investigates how the relative truthfulness increases based on these thresholds.
  - ``ex_4_a`` Initial results for the GAT architecture with ENS+MVE on the ClogP dataset.
  - ``ex_4_i`` Updated results using the "fixed" version of MVE using the GAT architecture with ENS+MVE on the ClogP dataset.