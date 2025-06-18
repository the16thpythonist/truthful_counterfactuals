import os
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from truthful_counterfactuals.metrics import threshold_error_reduction
from truthful_counterfactuals.metrics import counterfactual_truthfulness
from truthful_counterfactuals.metrics import threshold_counterfactual_truthfulness
from truthful_counterfactuals.metrics import nll_score, rll_score
from .util import ARTIFACTS_PATH


def test_threshold_error_reduction_basically_works():
    """
    There should be no error when computing the threshold_error_reduction function.
    """
    uncertainties = np.random.rand(100)
    errors = np.random.rand(100)
    
    num_bins = 25
    ths, rds = threshold_error_reduction(uncertainties, errors, num_bins=num_bins)
    
    assert len(ths) == num_bins
    assert len(rds) == num_bins


def test_threshold_error_reduction_perfect_correlation():
    """
    the ``threshold_error_reduction`` function should return an AUC value of 0.5 when the uncertainties
    and the errors are perfectly correlated.
    """
    uncertainties = np.random.uniform(0, 10, 100)
    errors = uncertainties
    
    num_bins = 25
    ths, rds = threshold_error_reduction(
        uncertainties, errors,
        num_bins=num_bins,
        error_func=np.mean,
    )
    
    # plotting the results for visual inspection
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
    ax.plot(ths, rds)
    ax.set_xlabel('uncertainty threshold')
    ax.set_ylabel('error reduction')
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_threshold_error_reduction_perfect_correlation.pdf')
    fig.savefig(fig_path)
    
    auc_value = auc(ths, rds)
    print(auc_value)
    assert np.isclose(auc_value, 0.5, atol=0.1)
    
    
def test_counterfactual_truthfulness_basically_works():
    """
    The ``counterfactual_truthfulness`` function should return a value between 0 and 1 which 
    describes the percentage of counterfactuals that are truthful aka where the predicted counterfactual 
    direction matches the true counterfactual direction.
    """
    counterfactual_datas = [
        {
            'org_pred': random.random(),
            'org_true': random.random(),
            'cf_pred': random.random(),
            'cf_true': random.random(),
        }
        for i in range(500)
    ]
    
    tr = counterfactual_truthfulness(
        counterfactual_datas=counterfactual_datas, 
        mode='regression'
    )
    assert isinstance(tr, float)
    assert 0 <= tr <= 1, "Truthfulness should be between 0 and 1"
    

def test_threshold_counterfactual_truthfulness_basically_works():
    """
    The ``threshold_counterfactual_truthfulness`` function should return the truthfulness values for
    different uncertainty thresholds (counterfactual uncertainty).
    """
    counterfactual_datas = [
        {
            'org_pred': random.random(),
            'org_true': random.random(),
            'cf_pred': random.random(),
            'cf_true': random.random(),
            'cf_uncertainty': random.random(),
        }
        for i in range(500)
    ]
    
    ths, trs, lens = threshold_counterfactual_truthfulness(
        counterfactual_datas=counterfactual_datas,
        mode='regression',
        num_bins=10,
    )
    
    assert isinstance(ths, np.ndarray), len(ths) == 10
    assert isinstance(trs, np.ndarray), len(trs) == 10
    assert isinstance(lens, np.ndarray), len(lens) == 10


def test_nll_score_basically_works():
    """
    The ``nll_score`` function should return a value without error.
    """
    y_true = np.random.rand(100)
    y_pred = np.random.rand(100)
    sigma_pred = np.random.rand(100) + 0.1  # to avoid division by zero
    
    nll = nll_score(y_true, y_pred, sigma_pred)
    
    assert isinstance(nll, float)


def test_rll_score_basically_works():
    """
    The ``rll_score`` function should return a value without error.
    """
    y_true = np.random.rand(100)
    y_pred = np.random.rand(100)
    sigma_pred = np.random.rand(100) + 0.1  # to avoid division by zero
    
    rll = rll_score(y_true, y_pred, sigma_pred)
    
    assert isinstance(rll, float)


def test_rll_score_good_vs_bad_uncertainty():
    """
    The ``rll_score`` function should return a lower value for a good uncertainty estimation compared to a bad one.
    """
    y_true = np.random.uniform(-10, 10, 100)
    y_pred = y_true + np.random.normal(0, 0.1, 100)  # good predictions
    
    # perfect uncertainty estimation
    sigma_pred_perf = np.abs(y_true - y_pred)
    # good uncertainty estimation
    sigma_pred_good = sigma_pred_perf + np.random.normal(0, 0.001, 100)  
    # bad uncertainty estimation
    sigma_pred_bad = sigma_pred_perf + np.random.normal(0, 0.1, 100)
    
    rll_perf = rll_score(y_true, y_pred, sigma_pred_perf)
    rll_good = rll_score(y_true, y_pred, sigma_pred_good)
    rll_bad = rll_score(y_true, y_pred, sigma_pred_bad)
    
    print(f"RLL Perfect: {rll_perf}, RLL Good: {rll_good}, RLL Bad: {rll_bad}")
    
    assert rll_good > rll_bad
