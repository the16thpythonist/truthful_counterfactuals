import os
import copy
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class Exponential(nn.Module):
    """
    Exponential torch activation function as a module.
    """
    def forward(self, x):
        x = torch.exp(x)
        x = x + 0 # I kid you not, this is actually necessary for the autograd to work...
        return x


class InitializationMixin:
    
    def __init__(self,
                 init_gain: float = 0.5,
                 **kwargs,
                 ) -> None:
        self.init_gain = init_gain
        self.hparams.update({
            'init_gain': init_gain,
        })
    
    def initialize_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=self.init_gain)


# == MEAN VARIANCE ESTIMATION ==

class MveCallback(pl.Callback):
    """
    A PyTorch Lightning callback for Mean-Variance Estimation (MVE) training.

    This callback enables variance training after a specified warmup period and applies gradient clipping
    to stabilize training. This callback only works for LightningModules which also implement the 
    ``MeanVarianceMixin``.

    :param warmup_fraction: Fraction of total epochs to use for warmup before enabling variance training.
    :type warmup_fraction: float
    :param clip_value: Value for gradient clipping to stabilize training.
    :type clip_value: float
    """
    
    def __init__(self, 
                 warmup_fraction: float = 0.5,
                 clip_value: float = 50.0
                 ):
        self.warmup_fraction: float = warmup_fraction
        self.clip_value: float = clip_value
        self.warmup_epochs: Optional[int] = None  # Will be set during training start

    def on_train_start(self, trainer, pl_module):
        """
        Called when the training starts. Calculates the number of warmup epochs based on the total epochs
        configured in the trainer.

        :param trainer: The trainer instance.
        :type trainer: pl.Trainer
        :param pl_module: The model being trained.
        :type pl_module: pl.LightningModule
        """
        self.warmup_epochs = int(trainer.max_epochs * self.warmup_fraction)

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each training epoch. Enables variance training and applies gradient clipping
        when the warmup period is over.

        :param trainer: The trainer instance.
        :type trainer: pl.Trainer
        :param pl_module: The model being trained.
        :type pl_module: pl.LightningModule
        """
        if trainer.current_epoch == self.warmup_epochs:
            
            print('Switching to mean-variance training...')
            # Since the module has to implement the MeanVarianceMixin, this method switches on the 
            # loss calculation for the variance and enables the variance training.
            pl_module.enable_variance_training()
            trainer.gradient_clip_val = self.clip_value
            trainer.gradient_clip_algorithm = 'value'


class MeanVarianceMixin:
    """
    A mixin that implements the basis for Mean-Variance Estimation (MVE) training of a AbstractGraphModel.
    
    **Mixin Conditions**
    
    The following conditions have to be fulfilled by the base model sclass for this mixin to work properly:
    
    - The model must have a forward method that returns a dictionary with the key 
      "graph_embedding" which is the vector that will be used as the basis of the 
      variance estimation.
    - The model must have a "self.embedding_dim" attribute that contains the dimensionality
      of the graph embedding vector which is the result of the graph encoder subnetwork.
    
    **Implementation**
    
    Generally MVE requires some kind of additional network structure that directly predicts a variance value 
    on top of the actual target value prediction. The constructor of this mixin dynamically adds such a network 
    which predicts the variance from an MLP on the graph embedding.
    
    After the ``enable_variance_training`` method is called, the model will use a custom training step that
    includes the variance loss in addition to the prediction loss.
    
    The variance prediction can be obtained with the ``predict_variance`` method which accepts a graph embedding 
    tensor and returns a tensor with the predicted variance values.
    """
    def __init__(self,
                 variance_units: list[int] = [32, 1],
                 beta: float = 0.5,
                 **kwargs
                 ):
        
        self.epsilon = 1e-6
        self.variance_units = variance_units
        self.beta = beta
        
        self.hparams.update({
            'variance_units': variance_units,
        })
        
        prev_features = self.embedding_dim
        self.variance_layers = torch.nn.ModuleList([])
        for i, units in enumerate(variance_units):
            
            if i + 1 == len(variance_units):
                # paper recommends the last layer of the variance units to be exponential 
                # activation function to ensure positive variance
                lay = torch.nn.Sequential(
                    nn.Linear(prev_features, units),
                    Exponential(),
                )
                
            else:
                lay = torch.nn.Sequential(
                    nn.Linear(prev_features, units),
                    nn.LeakyReLU(),
                )
                
            prev_features = units
            self.variance_layers.append(lay)
            
    def predict_variance(self,
                         graph_embedding: torch.Tensor,
                         ) -> torch.Tensor:
        
        variance = graph_embedding
        for lay in self.variance_layers:
            variance = lay(variance)
        
        return variance
    
    def mve_loss(self, 
                 out_true: torch.Tensor, 
                 out_pred: torch.Tensor,
                 out_var: torch.Tensor,
                 ) -> torch.Tensor:
        
        loss = 0.0
        out_var += self.epsilon

        #loss += 0.5 * torch.mean(torch.log(out_var) + (out_true - out_pred) ** 2 / out_var)
    
        out_std = torch.sqrt(out_var)
        loss += torch.mean(0.5 * out_std.detach()**(2 * self.beta) * (torch.log(out_var) + (out_true - out_pred) ** 2 / out_var))
    
        return loss
    
    def _training_step(self, data: torch.Tensor) -> torch.Tensor:
        
        batch_size = np.max(data.batch.detach().cpu().numpy()) + 1
        info = self.forward(data)
        
        out_pred = info['graph_output']
        out_true = data.y.view(out_pred.shape)
        
        graph_embedding = info['graph_embedding']
        out_var = self.predict_variance(graph_embedding)
        
        loss = self.mve_loss(out_true, out_pred, out_var)
        self.log('mve_loss', loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        
        return loss
    
    def enable_variance_training(self):
        
        #setattr(self, '_training_step', self.training_step)
        setattr(self, 'training_step', self._training_step)
        
    def configure_callbacks(self):
        return super().configure_callbacks()
        return super().configure_callbacks() + [
            MveCallback(),
        ]


# == STOCHASTIC WEIGHT AVERAGING GAUSSIAN ==
        
class SwagCallback(pl.Callback):
    
    def __init__(self, 
                 epoch_start: int = None,
                 **kwargs,
                 ) -> None:
        
        self.epoch_start = epoch_start
        
        self.epoch_count = 0
        
    def on_train_epoch_end(self, trainer, pl_module):
        
        self.epoch_count += 1
        
        if (self.epoch_start is not None) and (self.epoch_count >= self.epoch_start):
            
            # This method will store a copy of all the weights of the model in an additional list 
            # within the module itself.
            print('recording weight snapshot...')
            pl_module.record_snapshot()
        
    def on_train_start(self, trainer, pl_module):
        
        self.epoch_count = 0
        
    def on_train_end(self, trainer, pl_module):
        
        if len(pl_module.snapshot_list) != 0:
            # At the end of the training, this method uses all the previously recorded weight snapshots and 
            # calculates the distribution of the weights and stores the information about the distribution
            # as additional properties of the module.
            print('Calculating the weight distribution...')
            pl_module.calculate_distribution()


class SwagMixin:
    
    def __init__(self, 
                 epoch_start: int = None,
                 **kwargs):
        
        self.epoch_start = epoch_start
        
        self.snapshot_list: list[nn.ParameterList] = []
        
        self.weights_mean = None
        self.weights_std = None

    def record_snapshot(self):
        
        parameter_list = nn.ParameterList()
        for param in self.parameters():
            parameter_list.append(param.detach().clone())
        
        self.snapshot_list.append(parameter_list)
    
    def calculate_distribution(self):
        
        assert len(self.snapshot_list) > 2, 'Need at least 3 snapshots to calculate the weight distribution'
        
        snapshot_weights = []
        for param_list in self.snapshot_list:
            weights = parameters_to_vector(param_list)
            snapshot_weights.append(weights)
            
        snapshot_weights = torch.stack(snapshot_weights, dim=-1)
        
        self.weights_mean = torch.mean(snapshot_weights, dim=-1)
        self.weights_std = torch.std(snapshot_weights, dim=-1)
    
    def sample(self):
        
        with torch.no_grad():
            model = copy.copy(self)
            sample = torch.normal(self.weights_mean, self.weights_std).to(model.device)
            vector_to_parameters(sample, model.parameters())
    
        return model
    
    def configure_callbacks(self):
        return super().configure_callbacks() + [
            SwagCallback(epoch_start=self.epoch_start),
        ]
    
    # ~ saving and loading
    # For SWAG we need to maintain a history of the model's weights during the training process. This list 
    # of weights is integral to the swag method and therefore the following section defines custom saving 
    # and loading methods for the model that store and restore this list of weights.
    
    def additional_save(self) -> dict:
        return {
            'snapshot_list': self.snapshot_list,
            'weights_mean': self.weights_mean,
            'weights_std': self.weights_std,
        }
        
    @classmethod
    def load(cls, path: str):
        model = super().load(path)
        data = torch.load(path)

        model.snapshot_list = data['snapshot_list']
        model.weights_mean = data['weights_mean']
        model.weights_std = data['weights_std']
        
        return model