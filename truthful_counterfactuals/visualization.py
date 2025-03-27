import typing as typ

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def plot_threshold_error_reductions(ax:plt.Axes,
                                    thresholds: np.ndarray,
                                    reductions: np.ndarray,
                                    plot_reference: bool = True,
                                    color: str = 'orange',
                                    ):
    """
    Given a set of uncertainty ``thresholds`` and the corresponding error ``reductions``, this function will plot 
    the uncertainty error reduction curve onto the given axis ``ax``.
    
    Note that both the thresholds and reductions should be in the range [0, 1].
    
    :param ax: the matplotlib axis to plot the curve on
    :param thresholds: the uncertainty thresholds to plot as an array of shape (len, )
    :param reductions: the error reductions to plot as an array of shape (len, )
    :param plot_reference: whether to plot the reference line as well,
    :param color: the color of the curve to plot. Default is 'orange'.
    
    :return: the matplotlib axis with the plotted curve
    """
    if plot_reference:
        
        ax.plot(
            [0, 1], [0, 1], 
            color='gray', 
            linestyle='--', 
            alpha=1.0, 
            label='perfect corr.'
        )
    
    auc_value = auc(thresholds, reductions)
    ax.plot(thresholds, reductions, label=f'AUC: {auc_value:.3f}', color=color)
    ax.fill_between(thresholds, reductions, alpha=0.2, color=color)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return ax


def plot_threshold_truthfulness(ax: plt.Axes,
                                thresholds: np.ndarray,
                                truthfulness: np.ndarray,
                                lengths: typ.Optional[np.ndarray] = None,
                                color: str = 'purple',
                                color_lengths: str = 'gray',
                                ):
    
    th_min = np.min(thresholds)
    th_max = np.max(thresholds)
    ax.plot(thresholds, truthfulness, color=color)
    ax.set_xlim([th_min, th_max])
    ax.set_xlabel('rel. uncertainty threshold')
    ax.set_ylabel('truthfulness')
    
    ax_len = None
    if lengths is not None:
        
        len_max = np.max(lengths)
        lengths = lengths / len_max
        
        ax_len = ax.twinx()
        ax_len.plot(thresholds, lengths, ls='--', color=color_lengths)
        ax_len.set_ylabel('retention ratio')

    return ax, ax_len
