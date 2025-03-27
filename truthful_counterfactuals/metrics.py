from typing import List
import typing as typ

import numpy as np


def sign(value: float) -> float:
    
    if value >= 0:
        return +1.0
    else:
        return -1.0
    
    
def intervals_overlap(interval_1: tuple, interval_2: tuple) -> bool:
    
    min_1 = min(interval_1)
    max_1 = max(interval_1)
    
    min_2 = min(interval_2)
    max_2 = max(interval_2)
    
    return max(min_1, min_2) <= min(max_1, max_2)


def threshold_error_reduction(uncertainties: np.ndarray,
                              errors: np.ndarray,
                              error_func: typ.Callable = np.mean,
                              num_bins: int = 10,
                              percentile: int = 5,
                              ) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an array of ``uncertainties`` and an array of ``errors``, this function computes the error reduction
    for different uncertainty thresholds. The error reduction is computed as the relative error reduction
    compared to the total error. The function returns two arrays: the first array contains the relative uncertainty
    thresholds and the second array contains the corresponding relative error reductions for each threshold.
    
    :param uncertainties: An array of uncertainty values.
    :param errors: An array of true error values.
    :param error_func: A function that computes the cumulative error whose reduction is to be measured.
        default is np.mean (average error).
    :param num_bins: The number of bins to use for the uncertainty thresholds. The more bins, the more 
        fine-grained the uncertainty-error reduction curve will be
    :param percentile: The percentile margin to define the maximum and minimum uncertainty values inbetween
        which the thresholds will be computed by linear spacing according to the number of bins.
        
    :returns: two numpy arrays (uncertainty_thresholds, error_reductions) which together define the uncertainty-error
        reduction curve.
    """
    error_cum: float = error_func(errors)
    
    unc_min = np.percentile(uncertainties, percentile)
    unc_max = np.percentile(uncertainties, 100 - percentile)
    
    err_reductions = np.zeros(num_bins)
    #unc_thresholds = np.linspace(unc_min, unc_max, num_bins)
    unc_thresholds = np.linspace(unc_max, unc_min, num_bins)
    for index, th in enumerate(unc_thresholds):
        mask = uncertainties < th
        if len(errors[mask]) == 0:
            error = 0
        else:
            error = error_func(errors[mask])
        
        err_reductions[index] = (error_cum - error) / error_cum
        
    unc_thresholds = 1 - (unc_thresholds - unc_min) / (unc_max - unc_min)
    #unc_thresholds = (unc_thresholds - unc_min) / (unc_max - unc_min)
    
    return unc_thresholds, err_reductions
        
    
    
def counterfactual_truthfulness(counterfactual_datas: list[dict],
                                mode: typ.Literal['regression', 'classification'] = 'regression',
                                ) -> float:
    
    # For every counterfactual info we will store the boolean information about the truthfulness
    # in this list and then in end compute the global truthfulness value as the ratio of the True 
    # values to the total length of the list.
    values: list[bool] = []
    for info in counterfactual_datas:
        
        if mode == 'regression':
            # # We can simply check if difference of the prediction difference and ground truth value difference
            # # have the same sign to determine if they both go into the same direction.
            # # Multiplying the same signed values always is positive
            # sign_pred = sign(info['cf_pred'] - info['org_pred'])
            # sign_true = sign(info['cf_true'] - info['org_true'])
            # values.append(sign_pred * sign_true > 0)

            error_org = abs(info['org_pred'] - info['org_true'])
            interval_org = (info['org_pred'] - error_org, info['org_pred'] + error_org)
            
            error_cf = abs(info['cf_pred'] - info['cf_true'])
            interval_cf = (info['cf_pred'] - error_cf, info['cf_pred'] + error_cf)
            
            values.append(not intervals_overlap(interval_org, interval_cf))
        
    if len(values) == 0:
        return 0.0
        
    return len([value for value in values if value]) / len(values)


def threshold_counterfactual_truthfulness(counterfactual_datas: list[dict],
                                          mode: typ.Literal['regression', 'classification'] = 'regression',
                                          num_bins: int = 10,
                                          percentile: int = 2,
                                          ) -> float:
    
    uncertainties = np.array([info['cf_uncertainty'] for info in counterfactual_datas])
    
    unc_min = np.percentile(uncertainties, percentile)
    unc_max = np.percentile(uncertainties, 100 - percentile)
    
    unc_thresholds = np.linspace(unc_max, unc_min, num_bins)
    truthfulnesses = []
    lengths = []
    for index, th in enumerate(unc_thresholds):
        
        # we only want those elements that have a lower predicted uncertainty than the current 
        # threshold value. On those elements we are going to evaluate the truthfulness.
        threshold_infos = [info for info in counterfactual_datas if info['cf_uncertainty'] <= th]
        truthfulness = counterfactual_truthfulness(threshold_infos, mode=mode)
        truthfulnesses.append(truthfulness)
        lengths.append(len(threshold_infos))
        
    truthfulnesses = np.array(truthfulnesses)
    lengths = np.array(lengths)
        
    unc_thresholds = 1 - (unc_thresholds - unc_min) / (unc_max - unc_min)
        
    return unc_thresholds, truthfulnesses, lengths


def negative_log_likelihood(y_true: float,
                            y_pred: float,
                            sigma: float,
                            ) -> float:
    pred = (y_true - y_pred) ** 2 / (2 * sigma**2)
    dist = 0.5 * np.log(np.sqrt(2 * np.pi) * sigma**2)

    return dist + pred


def nll_score(y_true: np.ndarray, 
              y_pred: np.ndarray, 
              sigma_pred: np.ndarray
              ) -> float:
    """
    Computes the negative log likelihood (NLL) score for a given set of true values ``y_true``, predicted 
    values ``y_pred`` and the predicted uncertainties ``sigma_pred``. The negative log likelihood
    score is computed as the average of the negative log likelihoods of the Gaussian distribution
    with the predicted mean and variance for each true value. The score gives an indication of 
    how likely it is that the two sets of values (true and predicted) are drawn from the same
    distribution.
    
    :param y_true: Array of true values.
    :param y_pred: Array of predicted values.
    :param sigma_pred: Array of predicted uncertainties.
    
    :returns: The negative log likelihood score.
    """
    pred = (y_true - y_pred) ** 2 / (2 * sigma_pred)
    dist = 0.5 * np.log(np.sqrt(2 * np.pi) * sigma_pred)
    
    return np.mean(pred + dist)


def rll_score(y_true: np.ndarray,
              y_pred: np.ndarray,
              sigma_pred: np.ndarray,
              ) -> float:
    """
    Computes the *relative negative log likelihood* (RLL) score for a given set of true values ``y_true``,
    predicted values ``y_pred`` and the predicted uncertainties ``sigma_pred``.
    
    :param y_true: Array of true values.
    :param y_pred: Array of predicted values.
    :param sigma_pred: Array of predicted uncertainties.
    
    :returns: The relative negative log likelihood score.
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    nomin_values: List[float] = []
    denom_values: List[float] = []
    for y_t, y_p, sigma in zip(y_true, y_pred, sigma_pred):
        
        nomin_values.append((
            negative_log_likelihood(y_t, y_p, sigma) - 
            negative_log_likelihood(y_t, y_t, rmse)
        ))
        denom_values.append((
            negative_log_likelihood(y_t, y_p, np.abs(y_t - y_p)) -
            negative_log_likelihood(y_t, y_t, rmse)
        ))
        
    return np.sum(nomin_values) / np.sum(denom_values)               