"""
Base experiment implementation for the uncertainty quantification experiments. 

Note that in this base implementation the model training and the uncertainty quantification
are implemented as mock placeholders.

The experiment starts by loading the dataset of graphs and then creating a random 
train test split from it. After that the actual predictive model is trained to 
predict the target graph property. The model is then evaluated on the test set 
regarding its primary prediction performance. Afterwards, a separate uncertainty
estimator is used to quantify the uncertainty of the model predictions. The
uncertainty values are then evaluated by comparing them to the actual model errors
and plotting the uncertainty distribution as well as the uncertainty versus error
plots. Optionally, the uncertainty values can be calibrated using a separate
calibration set. The results of the experiment are saved as JSON artifacts and
the evaluation plots are saved as PDF files.
"""
import os
import time
import json
import random
from typing import List, Union

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from rich.pretty import pprint
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import auc
from sklearn.isotonic import IsotonicRegression
from torch_geometric.loader import DataLoader
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.util import dynamic_import
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader

from truthful_counterfactuals.utils import EXPERIMENTS_PATH
from truthful_counterfactuals.utils import np_array
from truthful_counterfactuals.models import AbstractGraphModel, MockModel
from truthful_counterfactuals.models import GINModel, GATModel, GCNModel
from truthful_counterfactuals.uncertainty import AbstractUncertainty, MockUncertainty
from truthful_counterfactuals.data import data_list_from_graphs
from truthful_counterfactuals.metrics import threshold_error_reduction
from truthful_counterfactuals.metrics import rll_score
from truthful_counterfactuals.visualization import plot_threshold_error_reductions

# == EXPERIMENT PARAMETERS ==

# :param IDENTIFIER:
#       The identifier of the experiment. This can later be used to differentiate 
#       filter experiments belonging to a certain group.
IDENTIFIER: str = 'default'
# :param SEED:
#       The random seed to be used for the experiment. This ensures reproducibility
#       for example regarding the train-test split.
SEED: int = 42

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
NUM_TEST: int = 1000
# :param NUM_VAL:
#       The number of elements to be randomly sampled as the validation set. This set is optionally 
#       used to calibrate the uncertainty values.
NUM_VAL: int = 1000
# :param DATASET_TYPE:
#       The type of the dataset. This can be either 'regression' or 'classification'.
#       Currently only regression supported!
DATASET_TYPE: str = 'regression'
# :param TARGET_INDEX:
#       The index of the target value in the graph labels.
TARGET_INDEX: int = 0

# == MODEL PARAMETERS ==

# :param MODEL_TYPE:
#       The type of the model to be used for the experiment. This can be either 'GIN' or 'GAT'.
#       The model type determines the architecture of the model that is used for the experiment.
MODEL_TYPE: str = 'gat'
# :param ENCODER_UNITS:
#       The number of units to be used in the encoder part of the model. This essentially determines
#       the number of neurons in each layer of the message passing encoder subnetwork.
ENCODER_UNITS = [64, 64, 64]
# :param PREDICTOR_UNITS:
#       The number of units to be used in the predictor part of the model. This essentially determines
#       the number of neurons in each layer of the final prediction subnetwork.
PREDICTOR_UNITS = [64, 32, 1]

# == TRAINING PARAMETERS == 

# :param LEARNING_RATE:
#       The learning rate to be used for the model training. Determines how much the model 
#       weights are updated during each iteration of the training.
LEARNING_RATE: float = 1e-5
# :param EPOCHS:
#       The number of epochs that the model should be trained for.
EPOCHS: int = 20
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

# == VISUALIZATION PARAMETERS ==

# :param FIG_SIZE:
#       The size of the figures that are generated during the experiment.
#       This value will be used both the width and the height of the plots.
#       This essentially determines the aspect ratio of how large the various text 
#       elements will appear within the plot.
FIG_SIZE: int = 5

__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

# ~ DATASET HOOKS


# ~ MODEL HOOKS

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


@experiment.hook('train_model__gcn', default=True, replace=False)
def train_model__gcn(e: Experiment,
                     index_data_map: dict,
                     train_indices: List[int],
                     test_indices: List[int],
                     **kwargs,
                     ) -> AbstractGraphModel:
    
    graphs_train = [index_data_map[index]['metadata']['graph'] for index in train_indices]
    loader_train = DataLoader(data_list_from_graphs(graphs_train), batch_size=e.BATCH_SIZE, shuffle=True)

    model = GCNModel(
        node_dim=e['node_dim'],
        edge_dim=e['edge_dim'],
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


@experiment.hook('train_model', default=True, replace=False)
def train_model(e: Experiment,
                index_data_map: dict[int, dict],
                train_indices: list[int],
                test_indices: list[int],
                ) -> AbstractGraphModel:
    """
    This hook is meant to train a new model given the information about the dataset contained in the 
    ``index_data_map`` and the information about the train/test split in ``train_indices`` and 
    ``test_indices``. The hook should return a fully trained model which implements the 
    AbstractGraphModel interface.
    
    ---
    
    Note that this default implementation returns a MockModel instance which determines the output 
    labels as the mean of the input labels. This is just a placeholder implementation and should be
    replaced with a proper model training procedure.
    """
    
    # We dynamically choose the specific hook to invoke based on the string model type specified 
    # as a parameter. This allows us to easily switch between different model types without having
    # to change the code in multiple places.
    model_type = e.MODEL_TYPE.lower()
    e.log(f'training model of type "{model_type}"...')
    return e.apply_hook(
        f'train_model__{model_type}',
        index_data_map=index_data_map,
        train_indices=train_indices,
        test_indices=test_indices,    
    )


@experiment.hook('uncertainty_estimator', default=True, replace=False)
def uncertainty_estimator(e: Experiment,
                          model: AbstractGraphModel
                          ) -> AbstractUncertainty:
    """
    This hook constructs the concrete AbstractUncertainty instance that can be used 
    to quantify the uncertainty of the model predictions. This hook should return an 
    instance which implements the AbstractUncertainty interface.
    
    ---
    
    This implementation returns a MockUncertainty instance with the given model and 
    dataset type.
    """
    return MockUncertainty(model=model, mode=e.DATASET_TYPE)


@experiment.hook('quantify_uncertainty', default=True, replace=False)
def quantify_uncertainty(e: Experiment,
                         model: AbstractGraphModel,
                         uncertainty_estimator: AbstractUncertainty,
                         index_data_map: dict[int, dict],
                         indices: list[int],
                         train_indices: List[int] = [],
                         ) -> list[dict]:
    """
    This hook quantifies the uncertainty of the model predictions. This hook should return
    a list of dictionaries, where each dictionary contains the results of the uncertainty 
    quantification for a single graph.
    
    ---
    
    This implementation evaluates the uncertainty for each graph in the given indices 
    using the provided uncertainty estimator.
    """
    graphs = [index_data_map[index]['metadata']['graph'] for index in indices]
    results: list[dict] = uncertainty_estimator.evaluate_graphs(graphs)

    # for result in results:
    #     result['uncertainty'] = min(100., result['uncertainty'])

    return results


@experiment.hook('calibrate_uncertainty', default=True, replace=False)
def calibrate_uncertainty(e: Experiment,
                          model: AbstractGraphModel,
                          uncertainty_estimator: AbstractUncertainty,
                          index_data_map: dict[int, dict],
                          indices: list[int],
                          ) -> list[dict]:
    """
    This hook calibrates the uncertainty values. Internally, this hook modifies the 
    uncertainty_estimator instance such that all subsequent queries of the uncertainty 
    are calibrated to the scale determined by the model error on the given validation dataset.
    
    ---
    
    This implementation first predicts the uncertainties on the calibration set and 
    calculates the true uncertainties as the mean absolute errors. It then calibrates 
    the uncertainty estimator using these values.
    """
    
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
    

@experiment.hook('evaluate_model', default=True, replace=False)
def evaluate_model(e: Experiment, 
                   model: AbstractGraphModel,
                   index_data_map: dict,
                   indices: list[int],
                   identifier: str,
                   key: str = 'test',
                   **kwargs,
                   ) -> None:
    """
    This hook evaluates the model performance on a given set of indices. It logs the 
    evaluation metrics and generates plots for visualization.
    
    ---
    
    This implementation calculates the R2 and MAE metrics for regression tasks, logs 
    these values, and generates regression plots for visualization.
    """
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
        mse_value = mean_squared_error(out_true, out_pred)
        mae_value = mean_absolute_error(out_true, out_pred)
        e[f'{key}/metrics/r2'] = r2_value
        e[f'{key}/metrics/mse'] = mse_value
        e[f'{key}/metrics/mae'] = mae_value
        e.log(f' * r2: {r2_value:.3f}')
        e.log(f' * mae: {mae_value:.3f}')
        
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


@experiment.hook('evaluate_uncertainty', default=True, replace=False)
def evaluate_uncertainty(e: Experiment,
                         true_values: Union[List[float], np.ndarray],
                         pred_values: Union[List[float], np.ndarray],
                         uncertainties: Union[List[float], np.ndarray],
                         key: str = 'default',
                         **kwargs,
                         ) -> None:
    
    true_values = np.array(true_values)
    pred_values = np.array(pred_values)
    uncertainties = np.array(uncertainties)
    
    predictions = pred_values
    targets = true_values
    
    # The model error has to be calculated slightly differently depending on whether we are dealing 
    # with a regression or a classifiation problem.
    if e.DATASET_TYPE == 'regression':
        errors = [abs(float(true) - float(pred)) 
                  for true, pred in zip(true_values, pred_values)]
    elif e.DATASET_TYPE == 'classification':
        errors = [abs(pred - np.argmax(true))
                  for true, pred in zip(true_values, pred_values)]
    
    df = pd.DataFrame({
        'prediction':       predictions,
        'error':            errors,
        'uncertainty':      uncertainties,
    })
    
    # Replace NaN values with the mean of the column
    df = df.fillna(df.mean())
    upper = df['uncertainty'].quantile(0.95)
    
    # one interesting numerical metric is the correlation coefficient between the uncertainty and the error
    corr_value = df['uncertainty'].corr(df['error'])
    e[f'{key}/metrics/corr'] = corr_value
    e.log(f' * Correlation: {corr_value:.3f}')
    
    # Another interesting metric is the relative negative log likelihood (RLL) score which gives and 
    # indication how likely it is that the predicted values are drawn from the same distribution.
    rll_value = rll_score(np.array(targets), np.array(predictions), np.array(uncertainties))
    e[f'{key}/metrics/rll'] = rll_value
    e.log(f' * RLL: {rll_value:.3f}')
    
    # plotting uncertainty distribution
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(e.FIG_SIZE, e.FIG_SIZE))
    sns.histplot(data=uncertainties, kde=True, color='gray', ax=ax, bins=20)
    ax.set_title('Uncertainty Distribution')
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Frequency')
    e.commit_fig('uncertainty_distribution.pdf', fig)
    
    # plotting uncertainty versus model error
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(e.FIG_SIZE, e.FIG_SIZE))
    ax.scatter(errors, uncertainties, color='red', alpha=0.1, linewidths=0)
    ax.set_title('Uncertainty Error Correlation')
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Error')
    e.commit_fig('uncertainty_error.pdf', fig)
    
    quantile_val = df['uncertainty'].quantile(0.99)
    df_filtered = df[df['uncertainty'] <= quantile_val]
    g = sns.JointGrid(
        data=df_filtered,
        x='uncertainty',
        y='error',
    )
    g.plot_joint(sns.histplot, bins=50)
    g = g.plot_marginals(sns.histplot, edgecolor=None, bins=20, alpha=0.8, kde=True)
    
    # Add faint kde plot in the background
    sns.kdeplot(
        data=df_filtered,
        x='uncertainty',
        y='error',
        fill=True,
        alpha=0.15,  # Adjust alpha for faintness
        ax=g.ax_joint,  # Overlay on the joint plot
        color='gray',
    )
    
    g.figure.suptitle(f'Uncertainty Error Correlation: {corr_value:.3f}\n')
    g.figure.subplots_adjust(top=0.85)
    e.commit_fig('uncertainty_error.pdf', g.figure)
    
    # uncertainty threshold error reduction (mean)
    ths, rds = threshold_error_reduction(
        df['uncertainty'].values, df['error'].values,
        num_bins=50,
        error_func=np.mean,
    )
    auc_mean_value = auc(ths, rds)
    e[f'{key}/metrics/uer_auc_mean'] = auc_mean_value
    e.log(f' * UER-AUC (mean): {auc_mean_value:.3f}')
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(e.FIG_SIZE, e.FIG_SIZE))
    plot_threshold_error_reductions(ax, ths, rds, color='orange')
    ax.set_title(f'Threshold Error Reduction (mean)')
    ax.set_xlabel('relative uncertainty threshold')
    ax.set_ylabel('relative error reduction')
    ax.legend()
    e.commit_fig('threshold_error_reduction_mean.pdf', fig)
    
    # uncertainty threshold error reduction (median)
    ths, rds = threshold_error_reduction(
        df['uncertainty'].values, df['error'].values,
        num_bins=50,
        error_func=np.median,
    )
    auc_median_value = auc(ths, rds)
    e[f'{key}/metrics/uer_auc_median'] = auc_median_value
    e.log(f' * UER-AUC (median): {auc_median_value:.3f}')
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(e.FIG_SIZE, e.FIG_SIZE))
    plot_threshold_error_reductions(ax, ths, rds, color='green')
    ax.set_title(f'Threshold Error Reduction (median)')
    ax.set_xlabel('relative uncertainty threshold')
    ax.set_ylabel('relative error reduction')
    ax.legend()
    e.commit_fig('threshold_error_reduction_median.pdf', fig)

    # uncertainty threshold error reduction (max)
    ths, rds = threshold_error_reduction(
        df['uncertainty'].values, df['error'].values,
        num_bins=50,
        error_func=lambda x: np.max(x),
    )
    auc_max_value = auc(ths, rds)
    e[f'{key}/metrics/uer_auc_max'] = auc_max_value
    e.log(f' * UER-AUC (max): {auc_max_value:.3f}')
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(e.FIG_SIZE, e.FIG_SIZE))
    plot_threshold_error_reductions(ax, ths, rds, color='purple')
    ax.set_title(f'Threshold Error Reduction (max)')
    ax.set_xlabel('relative uncertainty threshold')
    ax.set_ylabel('relative error reduction')
    ax.legend()
    e.commit_fig('threshold_error_reduction_max.pdf', fig)


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    
    # ~ loading the dataset
    
    e.log('loading the dataset...')
    dataset_path: str = ensure_dataset(e.VISUAL_GRAPH_DATASET)
    dataset_name: str = os.path.basename(dataset_path)
    e['dataset_name'] = dataset_name
    e.log(f'using dataset "{dataset_name}"')
    
    reader = VisualGraphDatasetReader(dataset_path, logger=e.logger)
    index_data_map = reader.read()
    indices = list(index_data_map.keys())
    e.log(f'loaded {len(index_data_map)} elements')
    
    e.log('loading processing module...')
    processing_path = os.path.join(dataset_path, 'process.py')
    module = dynamic_import(processing_path)
    processing = module.processing
    e.log(f'loading processing: {processing}')
    
    example_graph = index_data_map[indices[0]]['metadata']['graph']
    e['node_dim'] = example_graph['node_attributes'].shape[1]
    e['edge_dim'] = example_graph['edge_attributes'].shape[1]

    target_dim = example_graph['graph_labels'].shape[0]
    e['target_dim'] = target_dim
    
    # We use the TARGET_INDEX parameter to select the target dimensionality of the graph labels 
    # only if there is actually ambiguity in the target dimensionality.
    if e.DATASET_TYPE == 'regression' and target_dim > 0:
        
        e.log(f'target dim: {target_dim} - selecting index: {e.TARGET_INDEX}')
    
        for index, data in index_data_map.items():
            # Selecting the target index
            t = e.TARGET_INDEX
            data['metadata']['graph']['graph_labels'] = data['metadata']['graph']['graph_labels'][t:t+1]

    
    # :hook after_dataset:
    #       The hook that is used to apply additional processing steps to the dataset after it has been loaded.
    #       It receives the index_data_map and the processing module as arguments and should not return anything.
    e.apply_hook(
        'after_dataset',
        index_data_map=index_data_map,
        processing=processing,
    )
    
    e.log('creating train-test split...')
    indices_train = indices
    
    # Setting the seed before doing the train test split for reproducibility.
    np.random.seed(e.SEED)
    random.seed(e.SEED)
    
    if e.TEST_INDICES_PATH is not None:
        e.log(f'loading test indices from file @ {e.TEST_INDICES_PATH}')
        with open(e.TEST_INDICES_PATH, 'r') as file:
            indices_test = json.load(file)
    
    else:
        e.log('sampling random test indices...')
        
        # NUM_TEST may be just the absolute integer number of test samples or a float 
        # fraction of the dataset size.
        num_test = e.NUM_TEST
        if isinstance(num_test, float):
            num_test = int(num_test * len(indices))
        
        indices_test = random.sample(indices_train, k=num_test)
    
    indices_train = list(set(indices_train) - set(indices_test))
    
    # NUM_VAL may be just the absolute integer number of validation samples or a float
    # fraction of the dataset size.
    num_val = e.NUM_VAL
    if isinstance(num_val, float):
        num_val = int(num_val * len(indices))
        
    indices_val = random.sample(indices_train, k=num_val)
    indices_train = list(set(indices_train) - set(indices_val))
    e['indices/train'] = indices_train
    e['indices/test'] = indices_test
    e['indices/val'] = indices_val
    e.log(f' * train: {len(indices_train)}')
    e.log(f' * test:  {len(indices_test)}')
    e.log(f' * val:   {len(indices_val)}')
    
    for index in indices:
        graph = index_data_map[index]['metadata']['graph']
        graph['graph_labels'] = graph['graph_labels'] + np.random.normal(0, 0.0, size=graph['graph_labels'].shape)
    
    # ~ starting the model training
    
    e.log('starting model training...')
    
    model: AbstractGraphModel = e.apply_hook(
        'train_model',
        index_data_map=index_data_map,
        train_indices=indices_train,
        test_indices=indices_test,
    )
    e.log(f'finished training  model')
    
    # ~ evaluating the models
    # In this step the model is just evaluated towards its primary objective, which is to predict the
    # target graph property. The evaluation is done on the test set and the results are saved as a JSON
    # artifact internally.
    
    e.log('evaluating the model...')
    e.log('test set evaluation...')
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
        
    # ~ uncertainty evaluation
    # After the model has been trained and evaluated, the uncertainty of the model predictions is quantified.
    # This is done by using the implementation of the uncertainty_estimator hook to quantify the uncertainty
    # based on the existing model. This yields some measure of uncertainty / standard deviation for each 
    # target value prediction.
    
    graphs = [index_data_map[index]['metadata']['graph'] for index in indices_test]
    infos = model.forward_graphs(graphs)
    
    # We need the graph output/prediction as part of the information attached to the graph.
    for graph, info in zip(graphs, infos):
        graph['graph_output'] = info['graph_output']
    
    e.log('constructing uncertainty estimator...')
    # :hook uncertainty_estimator:
    #       The hook that is used to construct the concrete AbstractUncertainty instance that can be used 
    #       to quantify the uncertainty of the model predictions. This hook should return an instance which 
    #       implements the AbstractUncertainty interface.
    uncertainty_estimator: AbstractUncertainty = e.apply_hook(
        'uncertainty_estimator',
        model=model,
        index_data_map=index_data_map,
        train_indices=indices_train,
    )
    
    e.log('quantifying uncertainty...')
    time_uncertainty_start = time.time()
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
    
    time_uncertainty_end = time.time()
    duration_uncertainty = time_uncertainty_end - time_uncertainty_start
    e['test/duration/uncertainty'] = duration_uncertainty
     
    for result, index, info in zip(results, indices_test, infos):
        
        metadata = index_data_map[index]['metadata']
        graph = metadata['graph']
        
        result['graph_labels'] = graph['graph_labels']
        result['index'] = index
        result['graph_output'] = info['graph_output']
        if 'graph_embedding' in info:
            result['graph_embedding'] = info['graph_embedding']
        if 'graph_repr' in graph:
            result['graph_repr'] = graph['graph_repr']
    
    # ~ uncertainty calibration
    # Optionally - before evaluating it - the uncertainty estimates can be calibrated with an external 
    # calibration set. This is done by comparing the predicted uncertainties with the true uncertainties
    # (the prediction error) on the calibration set and fitting a separate calibration model to adjust the
    # predicted uncertainties to better match the true uncertainties.
    
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
        
        for result, result_cal in zip(results, results_cal):
            result['uncertainty_raw'] = result['uncertainty']
            result['uncertainty'] = result_cal['uncertainty']
    
    # ~ evaluating the uncertainty
    # Finally we actually evaluate the uncertainty quantification. This is done by comparing the predicted
    # uncertainties with the actual model errors. The evaluation is done by calculating the correlation
    # between the two values and plotting the uncertainty distribution as well as the uncertainty versus
    # model error plots.
    
    graphs_test = [index_data_map[index]['metadata']['graph'] for index in indices_test]
    targets = [graph['graph_labels'] for graph in graphs_test]
    predictions = [result['prediction'] for result in results]
    uncertainties = [result['uncertainty'] for result in results]
    
    # Saving the raw values so they can be retrieved later on
    e['test/values/true'] = np.squeeze(np_array([graph['graph_labels'] for graph in graphs_test]))
    e['test/values/pred'] = np.squeeze(np_array([result['prediction'] for result in results]))
    e['test/values/sigma'] = np.squeeze(np_array([result['uncertainty'] for result in results]))
    
    e.log('evaluating uncertainty...')
    # :hook evaluate_uncertainty:
    #       This hook will evaluate the uncertainty quantification.
    e.apply_hook(
        'evaluate_uncertainty',
        true_values=targets,
        pred_values=predictions,
        uncertainties=uncertainties,
        key='test',
    )

    # ~ finalizing experiment
    e.log('experiment done...')


experiment.run_if_main()