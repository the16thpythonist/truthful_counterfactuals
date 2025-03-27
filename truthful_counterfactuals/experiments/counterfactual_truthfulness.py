import os
import time
import json
import random
import pathlib
from typing import List, Dict
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import rdkit.Chem as Chem
import pytorch_lightning as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import auc
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.data import VisualGraphDatasetWriter
from visual_graph_datasets.util import dynamic_import
from visual_graph_datasets.visualization.molecules import mol_from_smiles, visualize_molecular_graph_from_mol
from vgd_counterfactuals import CounterfactualGenerator
from vgd_counterfactuals.generate.molecules import get_neighborhood

from truthful_counterfactuals.utils import EXPERIMENTS_PATH
from truthful_counterfactuals.models import AbstractGraphModel
from truthful_counterfactuals.models import MockModel
from truthful_counterfactuals.data import DataLoader, data_list_from_graphs
from truthful_counterfactuals.metrics import threshold_error_reduction
from truthful_counterfactuals.metrics import counterfactual_truthfulness
from truthful_counterfactuals.metrics import threshold_counterfactual_truthfulness
from truthful_counterfactuals.uncertainty import AbstractUncertainty
from truthful_counterfactuals.uncertainty import MockUncertainty
from truthful_counterfactuals.visualization import plot_threshold_truthfulness
from truthful_counterfactuals.visualization import plot_threshold_error_reductions

mpl.use('Agg')

# == EXPERIMENT PARAMETERS ==

# :param SEED:
#       The seed to be used for all random number generators in the experiment.
SEED: int = 43
# :param IDENTIFIER:
#       The identifier of the experiment. This will be used to identify individual experiments 
#       belonging to the same experiment group.
IDENTIFIER: str = 'default'

# == SOURCE PARAMETERS ==

# :param VISUAL_GRAPH_DATASET:
#       The path to the visual graph dataset folder that should be used as the basis of the 
#       experiment.
VISUAL_GRAPH_DATASET: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'logp')
# :param TEST_INDICES_PATH:
#       The path to a file containing the indices of the test set. If this is set, the test set
#       will be loaded from this file instead of being randomly sampled. The file should be a 
#       single JSON list of integers.
TEST_INDICES_PATH: str = None
# :param NUM_TEST:
#       The number of elements to be randomly sampled as the test set. The uncertainty quantification 
#       as well as the final prediction performance will be evaluated on this test set at the end.
NUM_TEST: int = 50
# :param NUM_VAL:
#       The number of elements to be randomly sampled as the validation set. This set is optionally 
#       used to calibrate the uncertainty values.
NUM_VAL: int = 50
# :param NUM_TRAIN:
#       The number of elements to be used as the training set. The remaining elements will be used
#       as the test set.
NUM_TRAIN: int = 50
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
ENCODER_UNITS: list[int] = [64, 64, 64]
# :param PREDICTOR_UNITS:
#       The number of units to be used in the predictor part of the model. This essentially determines
#       the number of neurons in each layer of the final prediction subnetwork.
PREDICTOR_UNITS: list[int] = [32, 16, 1]

# == TRAINING PARAMETERS == 

# :param LEARNING_RATE:
#       The learning rate to be used for the model training. Determines how much the model 
#       weights are updated during each iteration of the training.
LEARNING_RATE: float = 1e-4
# :param EPOCHS:
#       The number of epochs that the model should be trained for.
EPOCHS: int = 10
# :param BATCH_SIZE:
#       The batch size to be used for the model training. Determines how many samples are
#       processed in each iteration of the training.
BATCH_SIZE: int = 32
# :param CALIBRATE_UNCERTAINTY:
#       Whether the uncertainty values should be calibrated using the validation set.
#       A calibration would mean that the predicted uncertainties and the true uncertainties 
#       (prediction error) are compared and a separate calibration model is fitted to 
#       adjust the predicted uncertainties to better match the true uncertainties.
CALIBRATE_UNCERTAINTY: bool = True

# == COUNTERFACTUAL PARAMETERS ==

# :param NUM_COUNTERFACTUAL_ORIGINALS:
#       The number of original elements from the test set for which counterfactuals should be generated.
NUM_COUNTERFACTUAL_ORIGINALS: int = 1
# :param NUM_COUNTERFACTUALS:
#       The number of counterfactuals to be generated from the neighborhood of each element within 
#       the test set.
NUM_COUNTERFACTUALS: int = 10

# == VISUALIZATION PARAMETERS ==

# :param FIG_SIZE:
#       The size of the figures that are generated during the experiment.
#       This value will be used both the width and the height of the plots.
#       This essentially determines the aspect ratio of how large the various text 
#       elements will appear within the plot.
FIG_SIZE: int = 5
# :param NUM_BINS:
#       The number of bins to be used for all calculations in the experiment that 
#       are based on some form of binning.
NUM_BINS: int = 100
# :param EXAMPLE_SMILES:
#       This is a list of SMILES strings which should be plotted as examples in the end of the 
#       experiment. These will be used as the basis for counterfactual generation and for 
#       each counterfactual it will also be determined whether they fall above or below the 
#       uncertainty threshold.
EXAMPLE_SMILES: List[str] = [
    'C1=CC=C(C=C1)C(=O)O',
    # butylbenzene
    'CCCCC1=CC=CC=C1',
    # caffeine
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    # aspirin
    'CC(=O)OC1=CC=CC=C1C(=O)O',
    # sulpiride
    'CCN1CCCC1CNC(=O)C2=C(C=CC(=C2)S(=O)(=O)N)OC',
    # benzodiazipine
    'C1=CC=C2C(=C1)C=CC=NN2',
    # phencyclidine
    'C1CCC(CC1)(C2=CC=CC=C2)N3CCCCC3',
    # 9bH-phenalen-1-ylmethanol
    'C1=CC2=CC=CC3=C(C=CC(=C1)C23)CO',
    # nicotine
    'CN1CCCC1c1cccnc1'
]

__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('train_model__gat', default=True, replace=False)
def train_model__gat(e: Experiment,
                     index_data_map: dict,
                     train_indices: List[int],
                     test_indices: List[int],
                     **kwargs,
                     ) -> AbstractGraphModel:
    
    graphs_train = [index_data_map[index]['metadata']['graph'] for index in train_indices]
    loader_train = DataLoader(data_list_from_graphs(graphs_train), batch_size=e.BATCH_SIZE, shuffle=True)
        
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
    
    # very important: At the end of the training we need to put the model into evaluation mode!
    model.eval()
    
    return model


@experiment.hook('train_model__gin', default=True, replace=False)
def train_model__gin(e: Experiment,
                     index_data_map: dict,
                     train_indices: List[int],
                     test_indices: List[int],
                     **kwargs,
                     ) -> AbstractGraphModel:
    
    graphs_train = [index_data_map[index]['metadata']['graph'] for index in train_indices]
    loader_train = DataLoader(data_list_from_graphs(graphs_train), batch_size=e.BATCH_SIZE, shuffle=True)
        
    model = GINModel(
        node_dim=e['node_dim'],
        edge_dim=e['edge_dim'],
        encoder_units=e.ENCODER_UNITS,
        predictor_units=e.PREDICTOR_UNITS,
        mode=e.DATASET_TYPE,
        learning_rate=e.LEARNING_RATE,
        epoch_start=int(0.8 * e.EPOCHS),
    )
        
    trainer = pl.Trainer(max_epochs=e.EPOCHS)
    trainer.fit(
        model,
        train_dataloaders=loader_train,
    )
    
    # very important: At the end of the training we need to put the model into evaluation mode!
    model.eval()
    
    return model

@experiment.hook('train_model', default=True, replace=False)
def train_model(e: Experiment,
                index_data_map: dict[int, dict],
                train_indices: list[int],
                test_indices: list[int],
                ) -> AbstractGraphModel:

    graphs_train = [index_data_map[index]['metadata']['graph'] for index in train_indices]
    loader_train = DataLoader(data_list_from_graphs(graphs_train), batch_size=e.BATCH_SIZE, shuffle=True)
        
    e.log('training mock model...')
        
    model = MockModel(
        node_dim=e['node_dim'],
        edge_dim=e['edge_dim'],
        out_dim=1,
    )
    
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS
    )
    trainer.fit(
        model,
        train_dataloaders=loader_train,
    )
    
    e.log(f' * model done')
    model.eval()
    
    return model


@experiment.hook('evaluate_model', default=True, replace=False)
def evaluate_model(e: Experiment, 
                   model: AbstractGraphModel,
                   index_data_map: dict,
                   indices: list[int],
                   identifier: str,
                   prefix: str = 'test',
                   ) -> None:
    e.log(f'evaluating model "{identifier}"...')

    graphs = [index_data_map[index]['metadata']['graph'] for index in indices]
    infos = model.forward_graphs(graphs)

    if e.DATASET_TYPE == 'regression':
        
        out_true = [float(graph['graph_labels']) for graph in graphs]
        out_pred = [float(info['graph_output']) for info in infos]
        df = pd.DataFrame({
            'true': out_true,
            'pred': out_pred,
        })
        
        r2_value = r2_score(out_true, out_pred)
        mae_value = mean_absolute_error(out_true, out_pred)
        e.log(f' * r2: {r2_value:.3f}')
        e.log(f' * mae: {mae_value:.3f}')
        e[f'r2/{prefix}'] = r2_value
        e[f'mae/{prefix}'] = mae_value
        
        # simple regression plot
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(e.FIG_SIZE, e.FIG_SIZE))
        sns.histplot(
            data=df,
            x='true',
            y='pred',
            ax=ax,
            bins=50,
        )
        ax.set_title(f'Regression Plot\n'
                     f'R2: {r2_value:.3f}, MAE: {mae_value:.3f}')
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        e.commit_fig(f'regression_{identifier}.pdf', fig)
        
        # advanced regression plot
        g = sns.JointGrid(
            data=df,
            x='true',
            y='pred',
        )
        g.figure.suptitle(f'Regression Plot\n'
                          f'R2: {r2_value:.3f}, MAE: {mae_value:.3f}')
        g.figure.subplots_adjust(top=0.85)
        g.ax_joint.plot([0, 1], [0, 1], color='gray', linestyle='--')
        g.plot_joint(sns.histplot, bins=50)
        g.plot_marginals(sns.histplot, edgecolor=None, bins=20, alpha=0.6, kde=True)
        
        e.commit_fig(f'regression_joint_{identifier}.pdf', g.figure)


@experiment.hook('uncertainty_estimator', default=True, replace=False)
def uncertainty_estimator(e: Experiment,
                          model: AbstractGraphModel
                          ) -> AbstractUncertainty:
    
    return MockUncertainty(model=model, mode=e.DATASET_TYPE)


@experiment.hook('quantify_uncertainty', default=True, replace=False)
def quantify_uncertainty(e: Experiment,
                         model: AbstractGraphModel,
                         uncertainty_estimator: AbstractUncertainty,
                         index_data_map: dict[int, dict],
                         indices: list[int],
                         ) -> list[dict]:
    
    graphs = [index_data_map[index]['metadata']['graph'] for index in indices]
    results: list[dict] = uncertainty_estimator.evaluate_graphs(graphs)

    return results


@experiment.hook('calibrate_uncertainty', default=True, replace=False)
def calibrate_uncertainty(e: Experiment,
                          model: AbstractGraphModel,
                          uncertainty_estimator: AbstractUncertainty,
                          index_data_map: dict[int, dict],
                          indices: list[int],
                          ) -> list[dict]:
    
    # ~ determine the uncertainty on the calibration set
    # as a first step we need to determine the predicted uncertainties on the calibration set
    # using the already existing hook implementation for that.
    # equally we can get the "true" uncertainty values from the dataset simply as the 
    # error (absolute difference between the prediction and gt value)
    
    e.log('predicting uncertainty on the calibration set...')
    graphs_cal = [index_data_map[index]['metadata']['graph'] for index in indices]
    results_cal = e.apply_hook(
        'quantify_uncertainty',
        model=model,
        uncertainty_estimator=uncertainty_estimator,
        index_data_map=index_data_map,
        indices=indices,
    )
    
    # The ground truth uncertainty values we will consider to be the mean absolute model errors.
    # AKA the difference between the model prediction and the actual true label.
    errors = [abs(float(result['prediction']) - float(graph['graph_labels'])) 
              for result, graph in zip(results_cal, graphs_cal)]
    errors = np.array(errors)
    
    # This method will actually fit the calibration mappers and store them as internal 
    # attributes of the uncertainty_estimator instance so that they can be used for 
    # subsequent uncertainty queries.
    uncertainty_estimator.calibrate(graphs_cal, errors)
    
    return None


@experiment.hook('evaluate_uncertainty', default=True, replace=False)
def evaluate_uncertainty(e: Experiment,
                         index_data_map: dict,
                         indices: list[int],
                         prefix: str = '',
                         ) -> dict[str, float]:
    """
    
    This hook expects that the index_data_map structure has been extended with the following 
    additional graph information:
    - graph_prediction: the prediction of the model for the given graph
    - graph_uncertainty: the predicted uncertainty of the model for the given graph
    
    returns a dictionary containing the aggregated uncertainty evaluation metrics for the given 
    indices.
    """
    
    graphs = [index_data_map[index]['metadata']['graph'] for index in indices]
    errors = np.array([abs(graph['graph_prediction'] - graph['graph_labels']) for graph in graphs])
    uncertainties = np.array([float(graph['graph_uncertainty']) for graph in graphs])
    df = pd.DataFrame({
        'error': np.squeeze(errors),
        'uncertainty': np.squeeze(uncertainties),
    })

    # First of all we want to determine the correlation coefficient between the predicted uncertainties 
    # and the error as the most simple evaluation metric for the qualitiy of the uncertainty.
    # one interesting numerical metric is the correlation coefficient between the uncertainty and the error
    corr_value = df['uncertainty'].corr(df['error'])
    e.log(f' * Correlation: {corr_value:.3f}')

    g = sns.JointGrid(
        data=df,
        x='uncertainty',
        y='error',
    )
    g.plot_joint(sns.histplot, bins=50)
    g = g.plot_marginals(sns.histplot, edgecolor=None, bins=20, alpha=0.8, kde=True)
    g.figure.suptitle(f'Uncertainty Error Correlation: {corr_value:.3f}\n')
    g.figure.subplots_adjust(top=0.85)
    e.commit_fig(f'{prefix}_uncertainty_error.pdf', g.figure)
    
    # The next step is to calculate the threshold error reduction for the given uncertainty values.
    # This is a metric that is used to determine how much the error can be reduced by filtering out
    # the elements with a higher uncertainty value.
    
    # uncertainty threshold error reduction (mean)
    ths, rds = threshold_error_reduction(
        df['uncertainty'].values, df['error'].values,
        num_bins=e.NUM_BINS,
        error_func=np.mean,
    )
    ths_mean, rds_mean = ths, rds
    
    auc_mean_value = auc(ths, rds)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(e.FIG_SIZE, e.FIG_SIZE))
    plot_threshold_error_reductions(ax, ths, rds, color='orange')
    ax.set_title(f'Threshold Error Reduction (mean)')
    ax.set_xlabel('relative uncertainty threshold')
    ax.set_ylabel('relative error reduction')
    ax.legend()
    e.commit_fig(f'{prefix}_threshold_error_reduction_mean.pdf', fig)

    # uncertainty threshold error reduction (max)
    ths, rds = threshold_error_reduction(
        df['uncertainty'].values, df['error'].values,
        num_bins=e.NUM_BINS,
        error_func=np.max,
    )
    ths_max, rds_max = ths, rds 
    
    auc_max_value = auc(ths, rds)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(e.FIG_SIZE, e.FIG_SIZE))
    plot_threshold_error_reductions(ax, ths, rds, color='purple')
    ax.set_title(f'Threshold Error Reduction (max)')
    ax.set_xlabel('relative uncertainty threshold')
    ax.set_ylabel('relative error reduction')
    ax.legend()
    e.commit_fig(f'{prefix}_threshold_error_reduction_max.pdf', fig)

    return {
        'corr': corr_value,
        'auc_mean': auc_mean_value,
        'auc_max': auc_max_value,
        # We also want to store the raw threshold and reduction values here for later analysis.
        # e.g. for plotting.
        '_thresholds_mean': ths_mean,
        '_reductions_mean': rds_mean,
        '_thresholds_max': ths_max,
        '_reductions_max': rds_max,
    }


@experiment.hook('query_target', default=True, replace=False)
def query_target(e: Experiment,
                 graph: dict,
                 ) -> np.ndarray:
    smiles = str(graph['graph_repr'].item()) if not isinstance(graph['graph_repr'], str) else graph['graph_repr']
    mol = Chem.MolFromSmiles(smiles)
    value = Chem.Crippen.MolLogP(mol)
    return np.array([value])


@experiment.hook('generate_neighborhood', default=True, replace=False)
def generate_neighborhood(e: Experiment,
                          index_data_map: dict[int, dict],
                          indices: list[int],
                          processing: MoleculeProcessing,
                          key: str = '',
                          **kwargs,
                          ) -> dict[int, dict]:
    """
    Given an ``index_data_map`` containing graph dict representations of molecular graphs, 
    a list of ``indices`` from this map to process and a MoleculeProcessing instance ``processing``,
    this method will generate the 1-edit graph neighborhood for all of those elements and return 
    a new index_data_map containing the original graphs and the neighborhood graphs. The indices 
    within this new map will be completely independent of the original indices. However, the 
    graphs will contain additional metadata fields "original_index" and "original_smiles" 
    pointing to the original index of the graph from which they were derived.
    
    :returns: index_data_map dict structure containing the newly generated neighborhood graphs.
    """
    
    path = os.path.join(e.path, f'neighborhood_{key}')
    os.mkdir(path)
    writer = VisualGraphDatasetWriter(path, chunk_size=10_000)
    
    # We will iterate over all the indices given via argument and for each corresponding input 
    # graph we will generate the neighboring graphs - first as smiles and then process them as 
    # the actual graph structures.
    # We'll use a VGD writer instance to actually write a new dataset to the experiment archive 
    # folder.
    cf_index = 0
    for c, index in enumerate(indices):
        graph = index_data_map[index]['metadata']['graph']

        smiles_org = graph['graph_repr']
        if isinstance(smiles_org, np.ndarray):
            smiles_org = str(graph['graph_repr'].item())
        
        neighborhood_data = get_neighborhood(
            smiles=smiles_org,
            fix_protonation=False,
        )
        
        for data in neighborhood_data:
            
            smiles = data['value']
            graph = processing.process(smiles)
            graph_labels = e.apply_hook(
                'query_target',
                graph=graph,
            )
            
            processing.create(
                value=smiles,
                index=str(cf_index),
                additional_graph_data={
                    'graph_labels': graph_labels,
                },
                additional_metadata={
                    'index': cf_index,
                    'original_smiles': smiles_org,
                    'original_index': index,
                },
                writer=writer,
            )
            cf_index += 1
            
        if c % 10 == 0:
            e.log(f' * processed {c}/{len(indices)} - {cf_index} neighbors')
    
    # In the end we can read the newly created dataset from the disk and then return the 
    # index data map structure that is returned by the read method of the reader instance.
    reader = VisualGraphDatasetReader(path)
    cf_index_data_map = reader.read()
    return cf_index_data_map


@experiment
def experiment(e: Experiment):
    
    e.log('starting the experiment...')
    
    # ~ loading the dataset
    
    dataset_path = e.VISUAL_GRAPH_DATASET
    e.log(f'loading the dataset @ {e.VISUAL_GRAPH_DATASET}')
    reader = VisualGraphDatasetReader(e.VISUAL_GRAPH_DATASET)
    index_data_map: dict[int, dict] = reader.read()
    indices = list(index_data_map.keys())
    e.log(f'loaded dataset with {len(index_data_map)} elements')

    e.log('loading processing module...')
    processing_path = os.path.join(dataset_path, 'process.py')
    module = dynamic_import(processing_path)
    processing = module.processing
    e.log(f'loading processing: {processing}')
    
    example_graph = index_data_map[indices[0]]['metadata']['graph']
    e['node_dim'] = example_graph['node_attributes'].shape[1]
    e['edge_dim'] = example_graph['edge_attributes'].shape[1]

    # :hook after_dataset:
    #       The hook that is used to apply additional processing steps to the dataset after it has been loaded.
    #       It receives the index_data_map and the processing module as arguments and should not return anything.
    e.apply_hook(
        'after_dataset',
        index_data_map=index_data_map,
        processing=processing,
    )
    
    e.log('creating train-test split...')
    random.seed(e.SEED)
    np.random.seed(e.SEED)
    
    indices_train = indices
    if e.TEST_INDICES_PATH is not None:
        e.log(f'loading test indices from file @ {e.TEST_INDICES_PATH}')
        with open(e.TEST_INDICES_PATH, 'r') as file:
            indices_test = json.load(file)
    
    else:
        # We want to support NUM_TEST to be either an absolute number of test elements or a 
        # fraction of the training set size.
        num_test = e.NUM_TEST
        if isinstance(num_test, float):
            num_test = int(len(indices_train) * num_test)
        
        e.log(f'sampling {num_test} random test indices...')
        indices_test = random.sample(indices_train, k=num_test)
    
    indices_train = list(set(indices_train) - set(indices_test))
    
    # We also want to support NUM_VAL to be either an absolute number or a fraction
    num_val = e.NUM_VAL
    if isinstance(num_val, float):
        num_val = int(num_val * len(indices_train))
    indices_val = random.sample(indices_train, k=num_val)
    indices_train = list(set(indices_train) - set(indices_val))
    if e.NUM_TRAIN is not None:
        num_train = e.NUM_TRAIN
        if isinstance(e.NUM_TRAIN, float):
            num_train = int(num_train * len(indices_train))
    
        indices_train = random.sample(indices_train, k=num_train)
    
    e.log(f' * train: {len(indices_train)}')
    e.log(f' * test:  {len(indices_test)}')
    e.log(f' * val:   {len(indices_val)}')
    e['indices/train'] = indices_train
    e['indices/test'] = indices_test
    e['indices/val'] = indices_val
    
    # ~ starting the model training
    
    e.log('starting model training...')
    
    time_start = time.time()
    # :hook train_model:
    #       The hook that is used to train the model. It receives the index_data_map, the train indices and 
    #       the test indices as arguments and should return the trained model which has to be a subclass of the 
    #       AbstractGraphModel base class. The hooks implementation should handle the instantiation of the model
    #       and the training process.
    model: AbstractGraphModel = e.apply_hook(
        'train_model',
        index_data_map=index_data_map,
        train_indices=indices_train,
        test_indices=indices_test,
    )
    e.log(f'finished training model after {time.time() - time_start:.2f}s')
    
    e.log('model forward pass on test set...')
    graphs_test = [index_data_map[index]['metadata']['graph'] for index in indices_test]
    infos_test = model.forward_graphs(graphs_test)
    
    e.log('updating index_data_map...')
    for info, index in zip(infos_test, indices_test):
        index_data_map[index]['metadata']['graph']['graph_prediction'] = info['graph_output']
    
    e.log('evaluating the model...')
    # :hook evaluate_model:
    #       The hook that is used to evaluate the model. It receives the model, the index_data_map, the indices
    #       of the test set and an identifier string as arguments. The identifier string is used to distinguish
    #       between different evaluation steps. The hook can be used to evaluate the model performance on the given 
    #       indices of the dataset.
    e.apply_hook(
        'evaluate_model',
        model=model,
        index_data_map=index_data_map,
        indices=indices_test,
        identifier='test',
    )
    
    e.apply_hook(f'saving model...')
    model_path = os.path.join(e.path, f'model.ckpt')
    model.save(model_path)
    
    # ~ uncertainty quantification
    
    # :hook uncertainty_estimator:
    #       The hook that is used to construct the concrete AbstractUncertainty instance that can be used 
    #       to quantify the uncertainty of the model predictions. This hook should return an instance which 
    #       implements the AbstractUncertainty interface.
    uncertainty_estimator: AbstractUncertainty = e.apply_hook(
        'uncertainty_estimator',
        model=model,
    )
    
    # :hook quantify_uncertainty:
    #       The hook that is used to quantify the uncertainty of the model predictions. This hook should return
    #       a list of dictionaries, where each dictionary contains the results of the uncertainty quantification
    #       for a single graph.
    results = e.apply_hook(
        'quantify_uncertainty',
        model=model,
        uncertainty_estimator=uncertainty_estimator,
        index_data_map=index_data_map,
        indices=indices_test,
    )
     
    for result, index, info in zip(results, indices_test, infos_test):
        
        metadata = index_data_map[index]['metadata']
        graph = metadata['graph']
        
        graph['graph_uncertainty'] = result['uncertainty']
        graph['graph_uncertainty_raw'] = result['uncertainty']
    
    # ~ uncertainty calibration
    
    if e.CALIBRATE_UNCERTAINTY:
        
        e.log('using the validation set to calibrate uncertainty...')
        
        # :hook calibrate_uncertainty:
        #       The hook that is used to calibrate the uncertainty values. Internally, this hook should modify the 
        #       uncertainty_estimator instance such that all subsequent queries of the uncertainty are calibrated 
        #       to the scale that is determined by the model error on the given validation dataset. The validation
        #       dataset is passed as a list of indices of the complete dataset.
        e.apply_hook(
            'calibrate_uncertainty',
            model=model,
            uncertainty_estimator=uncertainty_estimator,
            index_data_map=index_data_map,
            indices=indices_val,
        )
        
        e.log('determine calibrated uncertainties on the test set...')
        results_cal = e.apply_hook(
            'quantify_uncertainty',
            model=model,
            uncertainty_estimator=uncertainty_estimator,
            index_data_map=index_data_map,
            indices=indices_test,
        )
        
        for index, result_cal in zip(indices_test, results_cal):
            metadata = index_data_map[index]['metadata']
            graph = metadata['graph']
            
            graph['graph_uncertainty'] = result_cal['uncertainty']
    
    # ~ evaluate uncertainty
    # Here we want to evaluate the quality of the uncertatinty predictions on the test set itself to see if 
    # the uncertainty quantification is working as intended aka being correlated with the prediction error 
    # for example
    
    e.log('evaluating the uncertainty on the test set...')
    # :hook evaluate_uncertainty:
    #       This hook will evaluate the uncertainty predictions of the model on the given indices of the dataset.
    #       It should return a dictionary containing the aggregated evaluation metrics, but it also produces
    #       plots and other visualizations that are stored in the experiment archive.
    e['uncertainty/test'] = e.apply_hook(
        'evaluate_uncertainty',
        index_data_map=index_data_map,
        indices=indices_test,
        prefix='test',
    )
    
    # Additionally we need to determine a suitable uncertainty threshold now here on the test set! So we want 
    # to get one value of uncertainty which we can then later apply to filter the counterfactuals and evaluate 
    # the truthfulness increase, because this is how this method would be applied in the real-world: There we 
    # don't have ground truth target values for counterfactuals but at best only for the test set...
    # So we will take the uncertainty value at a certain percentile of the uncertainty distribution such 
    # that only a certain percentage of elements remains.
    uncertainties_test = np.array([index_data_map[index]['metadata']['graph']['graph_uncertainty'] for index in indices_test])
    uncertainty_threshold = np.percentile(uncertainties_test, 20)
    e['uncertainty_threshold'] = uncertainty_threshold
    e.log(f'uncertainty threshold: {uncertainty_threshold:.3f}')
     
    # ~ counterfactual generation
    
    # The first step in the counterfactual generation is to generate the complete neighborhood of the test set
    # So every feasible graph that is X graph edits away from one of the elements in the test set.
    
    indices_cf = random.sample(indices_test, k=e.NUM_COUNTERFACTUAL_ORIGINALS)
    
    e.log('generating neighborhood...')
    neighbor_index_data_map = e.apply_hook(
        'generate_neighborhood',
        index_data_map=index_data_map,
        processing=processing,
        indices=indices_cf,
    )
    e.log(f'generated {len(neighbor_index_data_map)} neighbors...')
    
    e.log('forward pass on neighborhood elements...')
    neighbor_graphs = [data['metadata']['graph'] for data in neighbor_index_data_map.values()]
    neighbor_infos = model.forward_graphs(neighbor_graphs)
    e.log('updating neighbor index_data_map...')
    for info, data in zip(neighbor_infos, neighbor_index_data_map.values()):
        data['metadata']['graph']['graph_prediction'] = info['graph_output']
    
    e.log('uncertainty quantification on neighborhood...')
    neighbor_results = e.apply_hook(
        'quantify_uncertainty',
        model=model,
        uncertainty_estimator=uncertainty_estimator,
        index_data_map=neighbor_index_data_map,
        indices=list(neighbor_index_data_map.keys()),
    )
    for result, data in zip(neighbor_results, neighbor_index_data_map.values()):
        data['metadata']['graph']['graph_uncertainty'] = result['uncertainty']
        
    # :hook evaluate_uncertainty:
    #       This hook will evaluate the uncertainty predictions of the model on the given indices of the dataset.
    #       It should return a dictionary containing the aggregated evaluation metrics, but it also produces
    #       plots and other visualizations that are stored in the experiment archive.
    e['uncertainty/neighbor'] = e.apply_hook(
        'evaluate_uncertainty',
        index_data_map=neighbor_index_data_map,
        indices=list(neighbor_index_data_map.keys()),
        prefix='neighbor',
    )
    
    neighbor_datas = [
        {
            'org_true': index_data_map[int(data['metadata']['original_index'])]['metadata']['graph']['graph_labels'],
            'org_pred': index_data_map[int(data['metadata']['original_index'])]['metadata']['graph']['graph_prediction'],
            'cf_true': data['metadata']['graph']['graph_labels'],
            'cf_pred': data['metadata']['graph']['graph_prediction'],
            'cf_uncertainty': data['metadata']['graph']['graph_uncertainty'],
            #'cf_uncertainty': data['metadata']['graph']['graph_labels'] - data['metadata']['graph']['graph_prediction'],
        }
        for data in neighbor_index_data_map.values()
    ]
    
    tr_value = counterfactual_truthfulness(
        counterfactual_datas=neighbor_datas,
        mode=e.DATASET_TYPE,
    )
    e.log(f'neighborhood truthfulness: {tr_value:.3f}')

    ths, trs, lens = threshold_counterfactual_truthfulness(
        counterfactual_datas=neighbor_datas,
        mode=e.DATASET_TYPE,
        num_bins=e.NUM_BINS,
    )
    e['truthfulness/neighbor'] = {
        'thresholds': ths,
        'truthfulness': trs,
        'lengths': lens,
    }
    # plot the threshold truthfulness values
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(e.FIG_SIZE, e.FIG_SIZE))
    ax.set_title('Neighbor Truthfulness')
    plot_threshold_truthfulness(
        ax=ax,
        thresholds=ths,
        truthfulness=trs,
        lengths=lens,
    )
    e.commit_fig('threshold_truthfulness.pdf', fig)
    
    # Previously, the truthfulness was evaluated on the full neighborhood of each test set 
    # molecule (all possible deviations). From the many neighbors that any individual element 
    # can have we now want to sample a subset of "counterfactuals" as those neighbors 
    # which have the highest predicted target value deviation! 
    e.log(f'sampling {e.NUM_COUNTERFACTUALS} from the neighborhood of each element...')

    counterfactual_index_data_map = {}
    original_keys = set(data['metadata']['original_index'] for data in neighbor_index_data_map.values())
    for original_key in original_keys:
        
        # This is the predicted target value of the original element from the test set that 
        # corresponds to the current original index.
        original_value: float = index_data_map[int(original_key)]['metadata']['graph']['graph_prediction']
        
        # These are all the neighbors belonging to the same original element from the test set 
        # as identified by the unique original index.
        neighbors = [
            data 
            for index, data in neighbor_index_data_map.items() 
            if data['metadata']['original_index'] == original_key
        ]
        # Now we want to sort them by the absolute deviation of the predicted target value
        # from the original element.
        neighbors = sorted(
            neighbors, 
            key=lambda d: abs(original_value - float(d['metadata']['graph']['graph_prediction'])),
            reverse=True,
        )
        
        # Finally, we add the top results to the list of actual counterfactuals.
        num = min(e.NUM_COUNTERFACTUALS, len(neighbors))
        for data in neighbors[:num]:
            counterfactual_index_data_map[data['metadata']['index']] = data
            
    e.log(f'generated {len(counterfactual_index_data_map)} counterfactuals...')
    
    e.log('evaluating performance on counterfactuals...')
    # :hook evaluate_model:
    #       The hook that is used to evaluate the model. It receives the model, the index_data_map, the indices
    #       of the test set and an identifier string as arguments. The identifier string is used to distinguish
    #       between different evaluation steps. The hook can be used to evaluate the model performance on the given
    #       indices of the dataset.
    e.apply_hook(
        'evaluate_model',
        model=model,
        index_data_map=counterfactual_index_data_map,
        indices=list(counterfactual_index_data_map.keys()),
        identifier='cf',
    )
    
    e.log('evaluating uncertainty on counterfactuals...')
    # :hook evaluate_uncertainty:
    #       This hook will evaluate the uncertainty predictions of the model on the given indices of the dataset.
    #       It should return a dictionary containing the aggregated evaluation metrics, but it also produces
    #       plots and other visualizations that are stored in the experiment archive.
    e['uncertainty/counterfactual'] = e.apply_hook(
        'evaluate_uncertainty',
        index_data_map=counterfactual_index_data_map,
        indices=list(counterfactual_index_data_map.keys()),
        prefix='counterfactual',
    )

    # Now we can do the same assessment of the truthfulness for these selected counterfactuals as well...
    # This includes for one thing the flat baseline truthfulness value without any modifications of the 
    # set of counterfactuals. Also includes plotting the threshold truthfulness values.

    counterfactual_datas = [
        {
            'org_true': index_data_map[int(data['metadata']['original_index'])]['metadata']['graph']['graph_labels'],
            'org_pred': index_data_map[int(data['metadata']['original_index'])]['metadata']['graph']['graph_prediction'],
            'cf_true': data['metadata']['graph']['graph_labels'],
            'cf_pred': data['metadata']['graph']['graph_prediction'],
            'cf_uncertainty': data['metadata']['graph']['graph_uncertainty'],
            #'cf_uncertainty': data['metadata']['graph']['graph_labels'] - data['metadata']['graph']['graph_prediction'],
        }
        for data in counterfactual_index_data_map.values()
    ]
    tr_value = counterfactual_truthfulness(
        counterfactual_datas=counterfactual_datas,
        mode=e.DATASET_TYPE,
    )
    e.log(f'counterfactual truthfulness: {tr_value:.3f}')

    ths, trs, lens = threshold_counterfactual_truthfulness(
        counterfactual_datas=counterfactual_datas,
        mode=e.DATASET_TYPE,
        num_bins=e.NUM_BINS,
    )
    e['truthfulness/counterfactual'] = {
        'thresholds': ths,
        'truthfulness': trs,
        'lengths': lens,
    }
    # plot the threshold truthfulness values
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(e.FIG_SIZE, e.FIG_SIZE))
    ax.set_title('Counterfactual Truthfulness')
    plot_threshold_truthfulness(
        ax=ax,
        thresholds=ths,
        truthfulness=trs,
        lengths=lens,
    )
    e.commit_fig('counterfactual_truthfulness.pdf', fig)

    # ~ evaluating the threshold truthfulness increase
    # Here we use the uncertainty threshold that we determined earlier on the test set to filter out the 
    # counterfactuals and compare the resulting truthfulness with the truthfulness of the complete set.
    e.log('evaluating threshold truthfulness increase...')
    counterfactual_datas_th = [
        cf for cf in counterfactual_datas 
        if cf['cf_uncertainty'] <= e['uncertainty_threshold']
    ]
    counterfactual_truthfulness_th = counterfactual_truthfulness(
        counterfactual_datas=counterfactual_datas_th,
        mode=e.DATASET_TYPE,
    )
    e['counterfactual_truthfulness/org'] = tr_value
    e['counterfactual_truthfulness/thr'] = counterfactual_truthfulness_th
    e['counterfactual_truthfulness/diff'] = counterfactual_truthfulness_th - tr_value  # want: positive
    e.log(f' * truthfulness increase: {counterfactual_truthfulness_th - tr_value:.2f}'
          f' = {counterfactual_truthfulness_th:.2f} - {tr_value:.2f}')
    e.log(f' * number of remaining counterfactuals: {len(counterfactual_datas_th)} '
          f'({len(counterfactual_datas_th) / len(counterfactual_datas) * 100:.2f}%)')

    # ~ plotting example molecules
    # In addition to the samples from the dataset itself we also want to provide the possibility of 
    # evaluating the pipeline on some custom example molecules which don't necessarily need to be 
    # part of the dataset.
    # The idea here is to convert these elements into graph dicts, generate the counterfactuals for 
    # them, use the model to predict the target and the uncertainty and then plot all of the results 
    # into a figure.
    
    if e.EXAMPLE_SMILES:
        e.log(f'processing {len(e.EXAMPLE_SMILES)} example molecules...')
        examples_path = os.path.join(e.path, '.examples')
        os.mkdir(examples_path)

        processing: MoleculeProcessing
        ex_index_data_map: Dict[int, dict] = defaultdict(dict)
        for c, smiles in enumerate(e.EXAMPLE_SMILES):
            metadata: dict = processing.create(
                value=smiles,
                index=str(c),
                output_path=examples_path,
                additional_graph_data={
                    'graph_repr': smiles,
                }
            )
            graph = metadata['graph']
            graph['graph_repr'] = smiles
            ex_index_data_map[c]['metadata'] = metadata
            
        ex_index_data_map = dict(ex_index_data_map)
        # :hook generate_neighborhood:
        #       This hook is used to generate the neighborhood of the given indices of the dataset.
        #       It should return a new index_data_map structure containing the neighborhood graphs of 
        #       the input graphs.
        exn_index_data_map = e.apply_hook(
            'generate_neighborhood',
            index_data_map=ex_index_data_map,
            indices=list(ex_index_data_map.keys()),
            processing=processing,
            key='examples',
        )
        
        # joint index_data_map
        jnt_index_data_map = dict(enumerate(list(ex_index_data_map.values()) + list(exn_index_data_map.values())))
        jnt_index_data_map = {
            index: data 
            for index, data in jnt_index_data_map.items() 
            if 'metadata' in data and 'node_attributes' in data['metadata']['graph']
        }
        jnt_graphs = [data['metadata']['graph'] for data in jnt_index_data_map.values()]
        
        jnt_infos = model.forward_graphs(jnt_graphs)
        for info, data in zip(jnt_infos, jnt_index_data_map.values()):
            graph = data['metadata']['graph']
            graph['graph_prediction'] = info['graph_output']
            
        jnt_results = e.apply_hook(
            'quantify_uncertainty',
            model=model,
            uncertainty_estimator=uncertainty_estimator,
            index_data_map=jnt_index_data_map,
            indices=list(jnt_index_data_map.keys()),
        )
        
        for info, result, data in zip(jnt_infos, jnt_results, jnt_index_data_map.values()):
            data['metadata']['graph']['graph_prediction'] = info['graph_output']
            data['metadata']['graph']['graph_uncertainty'] = result['uncertainty']
        
        # plotting the results for each example molecule individually
        for c, data in ex_index_data_map.items():
            
            graph = data['metadata']['graph']
            #e.log(f'  graph prediction: {graph["graph_prediction"]:.3f}')
            
            # Now we can retrieve all the neighbors for the current graph by going through the previously 
            # generated neighborhood dataset and matching the original_index field.
            neighbors: List[dict] = [
                _data['metadata']['graph']
                for _index, _data in exn_index_data_map.items()
                if _data['metadata']['original_index'] == c and
                    'graph_prediction' in _data['metadata']['graph']
            ]
        
            # Then we select the 10 neighbors with the highest predicted target value deviation.
            num_counterfactuals = min(e.NUM_COUNTERFACTUALS, len(neighbors))
            counterfactuals = list(sorted(
                neighbors,
                key=lambda item: item['graph_prediction'] - graph['graph_prediction']
            ))[:num_counterfactuals]
            
            fig, rows = plt.subplots(
                ncols=num_counterfactuals + 1,
                nrows=1, 
                figsize=(e.FIG_SIZE * (num_counterfactuals + 1), e.FIG_SIZE),
                squeeze=False,
            )
            
            # plotting the original molecule
            ax_org = rows[0][0]
            ax_org.set_xticks([])
            ax_org.set_yticks([])
            ax_org.set_facecolor('#E6F0FA')
            ax_org.set_title(f'Original\n'
                             f'Pred: {str(np.round(graph["graph_prediction"], 2))}\n'
                             f'$\\sigma$: {graph["graph_uncertainty"]:.2f}')
            # plotting the molecular graph itself
            smiles_org = graph['graph_repr']
            if isinstance(smiles_org, np.ndarray):
                smiles_org = str(smiles_org.item())
            mol_org = mol_from_smiles(smiles_org)
            visualize_molecular_graph_from_mol(
                ax=ax_org,
                mol=mol_org,
                image_width=1000,
                image_height=1000,
            )
            
            for k, cf in enumerate(counterfactuals):
                ax = rows[0][k + 1]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Counterfactual {k}\n'
                             f'Pred: {str(np.round(cf["graph_prediction"], 2))} - '
                             f'True: {str(np.round(cf["graph_labels"], 2))}\n'
                             f'$\\sigma$: {cf["graph_uncertainty"]:.2f} / {uncertainty_threshold:.2f}')
                
                # red background if the uncertainty threshold is exceeded
                if cf['graph_uncertainty'] > uncertainty_threshold:
                    ax.set_facecolor('#FFD6D6')
                    
                # plot the molecular graph
                smiles_cf = cf['graph_repr']
                if isinstance(smiles_cf, np.ndarray):
                    smiles_cf = str(smiles_cf.item())
                mol_cf = mol_from_smiles(smiles_cf)
                visualize_molecular_graph_from_mol(
                    ax=ax,
                    mol=mol_cf,
                    image_width=1000,
                    image_height=1000,
                )
                
            e.commit_fig(f'example_{c}.pdf', fig)


experiment.run_if_main()