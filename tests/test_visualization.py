import os
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .util import ARTIFACTS_PATH
from truthful_counterfactuals.metrics import threshold_error_reduction, threshold_counterfactual_truthfulness
from truthful_counterfactuals.visualization import plot_threshold_error_reductions, plot_threshold_truthfulness


class TestPlots:
    """
    Tests the various plotting functions.
    """
    
    def test_regression_plot(self):
        """
        Tests the plotting of a regression plot, which is a correlation plot between the 
        true values of a regression problem and the predicted values
        """
        out_true = np.random.rand(1000, )
        out_pred = np.random.rand(1000, )
        df = pd.DataFrame({
            'true': out_true,
            'pred': out_pred,
        })
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
        g = sns.histplot(
            data=df,
            x='true',
            y='pred',
            ax=ax,
            bins=25,
            color='blue',
        )
        
        path = os.path.join(ARTIFACTS_PATH, 'test_regression_plot.pdf')
        fig.savefig(path)
        
    def test_plot_threshold_truthfulness(self):
        """
        The plot_threshold_truthfulness function should be able to plot the results of the uncertainty threshold 
        truthfulnesss analysis. The function should plot the function of the truthfulness over different threshold values 
        and also the relative number of elements that are retained for each threshold value.
        """
        # first of all we are going to generate some data for the plot, we will create some completely 
        # random data for the uncertainty and the predictions.
        datas = [
            {
                'org_true': random.random(),
                'org_pred': random.random(),
                'cf_true': random.random(),
                'cf_pred': random.random(),
                'cf_uncertainty': random.random()
            }
            for i in range(1000)
        ]
        ths, trs, lens = threshold_counterfactual_truthfulness(
            datas,
            num_bins=100,    
        )
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
        plot_threshold_truthfulness(
            ax=ax,
            thresholds=ths,
            truthfulness=trs,
            lengths=lens,
        )
        fig_path = os.path.join(ARTIFACTS_PATH, 'test_plot_threshold_truthfulness.pdf')
        fig.savefig(fig_path)
