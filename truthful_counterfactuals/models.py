import os
import math
import random
import typing as typ
from typing import TypeVar, Generic, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.utils import scatter

from truthful_counterfactuals.data import loader_from_graphs
from truthful_counterfactuals.utils import get_version
from truthful_counterfactuals.mixins import InitializationMixin
from truthful_counterfactuals.mixins import MeanVarianceMixin
from truthful_counterfactuals.mixins import SwagMixin
from truthful_counterfactuals.mixins import EvidentialMixin

ModelType = TypeVar('ModelType')


# == BASE CLASSES ==

class AbstractGraphModel(pl.LightningModule):
    
    def __init__(self,
                 learning_rate: float = 1e-3,
                 mode: typ.Literal['regression', 'classification'] = 'regression',
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.mode = mode
        
        self.hparams.update({
            'learning_rate': learning_rate,
            'mode': mode,
        })
        
        if self.mode == 'regression':
            self.criterion = nn.MSELoss()
        elif self.mode == 'classification':
            self.criterion = nn.CrossEntropyLoss()
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, data: Data) -> torch.Tensor:
        
        loss = 0.0
        batch_size = np.max(data.batch.detach().cpu().numpy()) + 1
        
        info: dict = self.forward(data)
        out_pred = info['graph_output']
    
        out_true = data.y.view(out_pred.shape)
        
        loss += self.criterion(out_pred, out_true)
        self.log('loss', loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        
        return loss
        
    def split_infos(self, info: dict, data) -> list[dict]:
        """
        Given a single ``info`` dict which is returned by the model's forward method and the ``data`` object 
        that represents the input graphs, this method will split the information into a list of dictionaries
        where each dictionary holds the information for one graph in the given batch.
        
        The method uses the string key names of the attributes in the given info dict to infer attribute shape. 
        Keys can start with the following prefixes:
        - node: it is assumed that the attribute maps to each node of the graph
        - edge: it is assumed that the attribute maps to each edge of the graph
        - graph: it is assumed that the attribute maps to the whole graph
        
        :returns: a list of dictionaries where each dict has the same keys as the given "info" dict but the 
            values are split into the corresponding graphs in the batch.
        """
        # This is the actual size of the CURRENT batch. Usually this will be the same as the given batch 
        # size, but if the number of graphs is not exactly divisible by that number, it CAN be different!
        _batch_size = np.max(data.batch.detach().numpy()) + 1
        
        results: list[dict] = []
        for index in range(_batch_size):
                
            node_mask = (data.batch == index)
            edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
            
            result = {}
            for key, value in info.items():
                
                # Here we apply a bit of magic: Depending on the prefix of the string name of the attribute 
                # we are going to dynamically treat the corresponding values (=tensors) differently, since 
                # the names tell us in regard to what graph element those values are defined. For example for 
                # "graph" global properties we do not have to do any different processing, we can simply get 
                # the element with the correct index from the tensor. However, for node and edge based attributes 
                # we first have to construct the appropriate access mask to aggregate the correct values from 
                # the tensor.
                
                if key.startswith('graph'):
                    result[key] = value[index].detach().numpy()
                    
                elif key.startswith('node'):
                    array = value[node_mask].detach().numpy()
                    result[key] = array
                    
                elif key.startswith('edge'):
                    array = value[edge_mask].detach().numpy()
                    result[key] = array
                
            results.append(result)
    
        return results
    
    def forward_graphs(self, graphs: list[dict], batch_size: int = 128) -> list[dict]:
        
        with torch.no_grad():
            
            loader = loader_from_graphs(graphs, batch_size=batch_size)
            
            # This will be the data structure that holds all the results of the inference process. Each element 
            # in this list will be a dictionary holding all the information for one graph in the given list of 
            # graphs - having the same order as that list.
            results: list[dict] = []
            for data in loader:
                # This is the actual size of the CURRENT batch. Usually this will be the same as the given batch 
                # size, but if the number of graphs is not exactly divisible by that number, it CAN be different!
                _batch_size = np.max(data.batch.detach().numpy()) + 1
                
                # This line ultimately invokes the "forward" method of the class which returns a dictionary structure 
                # that contains all the various bits of information about the prediction process.
                info: dict = self.forward(data)
                    
                results += self.split_infos(info, data)
                
        return results
    
    def forward_graph(self, graph: dict) -> dict:
        return self.forward_graphs([graph])[0]

    def state_dict(self, *args, **kwargs) -> dict:
        """Return a copy of the model state dictionary."""
        state = super().state_dict(*args, **kwargs)
        return {k: v.clone() for k, v in state.items()}
    
    @classmethod
    def load(cls, path: str) -> 'AbstractGraphModel':
        """
        Loads the model from a persistent CKPT path at the given absolute ``path``. Returns the 
        reconstructed model instance.
        
        :returns: model instance
        """
        try:
            model = cls.load_from_checkpoint(path)
            model.eval()
            return model
            
        except Exception as exc:
            raise exc
        
    def additional_save(self) -> dict[str, typ.Any]:
        """
        Can be optionally overwritten to provide additional elements that should be saved in the model's
        persistent file representation. This method should return a dict with key value pairs where 
        the keys are string names and the values should be torch serializable values.
        
        :returns: dict
        """
        return {}
    
    def save(self, path: str) -> None:
        """
        Saves the model as a persistent file to the disk at the given ``path``. The file will be a torch
        ckpt file which is in essence a zipped archive that contains the model's state dictionary and the 
        hyperparameters that were used to create the model. Based on this information, the model can later 
        be reconstructed and loaded.
        
        :param path: The absolute file path of the file to which the model should be saved.
        
        :returns: None
        """
        torch.save({
            'class': self.__class__.__name__,
            'state_dict': self.state_dict(),
            'hyper_parameters': self.hparams,
            'pytorch-lightning_version': pl.__version__,
            'version': get_version(),
            **self.additional_save(),
        }, path)
    
    
# == SUB CLASSES ==

class MockModel(SwagMixin, MeanVarianceMixin, AbstractGraphModel):
    """
    A simple model that implements the AbstractGraphModel interface. Mainly used for testing.
    """
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 out_dim: int,
                 **kwargs,
                 ):
        AbstractGraphModel.__init__(self)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.embedding_dim = node_dim
        
        self.hparams.update({
            'node_dim': node_dim,
            'edge_dim': edge_dim,
            'out_dim': out_dim,
        })
        
        self.lay_conv = GCNConv(node_dim, node_dim)
        self.lay_pool = SumAggregation()
        self.lay_linear = nn.Linear(node_dim, out_dim)
        
        MeanVarianceMixin.__init__(self, **kwargs)
        SwagMixin.__init__(self, **kwargs)
        
    def forward(self, data: Data):
        node_input, edge_input, edge_indices = data.x, data.edge_attr, data.edge_index
        node_embedding = self.lay_conv(node_input, edge_indices)
        graph_embedding = self.lay_pool(node_embedding, data.batch)
        out = self.lay_linear(graph_embedding)
        
        return {
            'graph_output': out,
            'graph_embedding': graph_embedding
        }
        
        
class EnsembleModel(AbstractGraphModel, Generic[ModelType]):
    
    def __init__(self,
                 models: list[AbstractGraphModel] = [],
                 **kwargs,
                 ):
        AbstractGraphModel.__init__(self, **kwargs)
        
        self.models = models

    def forward(self, data: Data):
        
        infos: list[dict] = []
        for model in self.models:
            info = model.forward(data)
            infos.append(info)

        outputs = torch.stack([info['graph_output'] for info in infos], axis=0)
        return {
            'graph_output': torch.mean(outputs, axis=0),
            'graph_variance': torch.std(outputs, axis=0),
        }
        
    def save(self, path: str) -> None:
        
        # An ensemble model is a collection of models itself and therefore we have to do the saving and 
        # loading differently. Specifically, we want to create a folder instead of a single model file 
        # and then in that folder we want to save the individual models.
        
        os.mkdir(path)
        
        for index, model in enumerate(self.models):
            model_path = os.path.join(path, f'model_{index}.ckpt')
            model.save(model_path)
    
    @classmethod
    def load(cls, path: str):
        
        models = []
        for name in os.listdir(path):
            model_path = os.path.join(path, name)
            model = MockModel.load(model_path)
            models.append(model)
            
        return cls(models=models)
        
        
class RepulsiveEnsembleModel(EnsembleModel):
    """
    A derivative of a normal ensemble model that implements "repulsive ensembles".
    """
    
    def __init__(self,
                 models: List[AbstractGraphModel] = [],
                 repulsive_factor: float = 0.1,
                 **kwargs
                 ):
        EnsembleModel.__init__(self, models=models, **kwargs)
        
        for model in models[1:]:
            model.set_encoder(models[0])
            
        self.modules = nn.ModuleList(models)
        
        self.num_models: int = len(self.models)
        self.repulsive_factor = repulsive_factor
        self.hparams.update({
            'repulsive_factor': repulsive_factor,
        })
    
    def training_step(self, data: Data) -> torch.Tensor:
        
        loss = 0.0
        batch_size = np.max(data.batch.detach().cpu().numpy()) + 1
        
        data.x.requires_grad = True
        
        # In this list we are going to store the individual predictions of each model in the ensemble.
        loss_pred = 0.0
        out_preds: List[torch.Tensor] = []
        out_grads: List[torch.Tensor] = []
        for c, model in enumerate(self.models):
            # As each model is supposed to implement the AbstractGraphModel interface, we can assume 
            # that the output here is a dict which contains the output prediction among other things.
            info: dict = model.forward(data)
            # out_pred: (batch_size, out_dim)
            out_pred: torch.Tensor = info['graph_output']
            out_preds.append(out_pred)
             # out_true: (batch_size, out_dim)
            out_true = data.y.view(out_pred.shape)
            
            # Compute the first principal component of the embedding vectors
            emb = info['graph_embedding']
            #print(c, emb)
            
            emb_centered = emb - emb.mean(dim=0, keepdim=True)
            u, s, v = torch.svd(emb_centered)
            pc = v[:, 0:3]

            inp = data.x
            inp = emb
            out_grad = torch.autograd.grad(out_pred.sum(), inp, create_graph=True)[0]
            #out_grads.append(out_grad)
        
            proj = torch.mm(out_grad, pc)
            #print(emb.shape, out_grad.shape, proj.shape)
            out_grads.append(proj)
        
            loss_pred += self.criterion(out_pred, out_true)
        
        # The prediction loss is just the normal MSE / CCE loss for all the individual models 
        # in the ensemble.
        self.log('loss_pred', loss_pred, prog_bar=True, on_epoch=True, batch_size=batch_size)
        loss += loss_pred / self.num_models
        
        # ~ repulsive loss
        # In addition to the prediction loss, we also want to compute a repulsive loss term
        # which promotes the models to have a pairwise different output function.
        loss_rep = 0.0
        # This is the scalar average error over the current mini batch
        # avg_error: (1, )
        avg_error = torch.stack([(out_pred - out_true).abs() for out_pred in out_preds])
        avg_error = avg_error.mean()
        self.log('average_error', avg_error, prog_bar=False, on_epoch=True, batch_size=batch_size)
        # This is a term proportional to the disagreement between all the models in the ensemble 
        # for each of the elements in the batch.
        # mdl_disagree: (batch_size, )
        mdl_disagree = torch.stack([(out_pred - out_true).abs().mean(dim=1) for out_pred in out_preds])
        mdl_disagree = mdl_disagree.mean(dim=0)
        # Finally, this will give a normalized value (roughly between 0 and 1) that tells us how 
        # much the models disagree on each element in the batch
        # alpha: (batch_size, )
        alpha = mdl_disagree / (avg_error ** 2)
        self.log('alpha_mean', alpha.mean(), prog_bar=False, on_epoch=True, batch_size=batch_size)
        self.log('alpha_max', alpha.max(), prog_bar=False, on_epoch=True, batch_size=batch_size)
        self.log('alpha_min', alpha.min(), prog_bar=False, on_epoch=True, batch_size=batch_size)
        # Then we can pick a random pair of models and compute the repulsive loss term for them.
        # This is done by computing the squared difference of the output predictions of the two models
        # and then multiplying that with the alpha value which is a measure of how much the models
        # disagree on that specific element. We want to maximize the disagreement between the models 
        # preferably on those elements which they already disagree on.
        i, j = random.sample(range(self.num_models), 2)
        # Since the gradients exist on the node level we cast the graph-level alpha values to the 
        # individual nodes.
        alpha_nodes = alpha.detach().unsqueeze(1)
        #alpha_nodes = alpha_nodes[data.batch]
        loss_rep -= (alpha_nodes * (out_grads[i] - out_grads[j]).abs()).mean()
        
        self.log('loss_rep', loss_rep, prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log('repulsive_factor', self.repulsive_factor, prog_bar=False, on_epoch=True, batch_size=batch_size)
        loss += self.repulsive_factor * loss_rep
        
        # We also want to log the overall loss value.
        self.log('loss', loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        
        return loss


class GINModel(EvidentialMixin, SwagMixin, MeanVarianceMixin, InitializationMixin, AbstractGraphModel):
    
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 encoder_units: list[int],
                 predictor_units: list[int],
                 hidden_units: int = 128,
                 **kwargs,
                 ):
        AbstractGraphModel.__init__(self)
        SwagMixin.__init__(self, **kwargs)
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.encoder_units = encoder_units
        self.predictor_units = predictor_units
        self.hidden_units = hidden_units
        
        self.hparams.update({
            'node_dim': node_dim,
            'edge_dim': edge_dim,
            'encoder_units': encoder_units,
            'predictor_units': predictor_units,
            'hidden_units': hidden_units,
        })
        
        prev_features = self.encoder_units[0]
        self.lay_embedd = nn.Linear(
            in_features=node_dim,
            out_features=prev_features,
        )
        
        self.encoder_layers = nn.ModuleList([])
        for units in self.encoder_units:
            lay = GINEConv(
                nn.Sequential(
                    nn.Linear(prev_features, hidden_units),
                    #nn.BatchNorm1d(hidden_units),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_units, units),
                    #nn.BatchNorm1d(units),
                    nn.LeakyReLU(),
                ),
                edge_dim=edge_dim,
            )
            prev_features = units
            self.encoder_layers.append(lay)
            
        # This is the dimension of the graph embedding vector which is the result of the final layer of the 
        # encoder subnetwork. We need to define this attribute as a condition for some of the Mixin's
        self.embedding_dim = prev_features
        
        # The pooling layer can be used to aggregate all of the node embeddings into a single graph embedding.
        self.lay_pool = SumAggregation()
            
        self.predictor_layers = nn.ModuleList([])
        for i, units in enumerate(self.predictor_units):
            
            if i + 1 >= len(self.predictor_units):
                lay = nn.Linear(prev_features, units)
            else:
                lay = nn.Sequential(
                    nn.Linear(prev_features, units),
                    #nn.LayerNorm(units),
                    nn.LeakyReLU(),
                )
            
            prev_features = units
            self.predictor_layers.append(lay)
            
        # ~ implementing mixins
        
        MeanVarianceMixin.__init__(self, **kwargs)
        
        EvidentialMixin.__init__(self, **kwargs)
        
        InitializationMixin.__init__(self, **kwargs)
        self.initialize_weights()
    
    def forward(self, data: Data):
        node_input, edge_input, edge_indices = data.x, data.edge_attr, data.edge_index
        
        node_embedding = self.lay_embedd(node_input)
        for lay in self.encoder_layers:
            node_embedding = lay(node_embedding, edge_indices, edge_input)
            
        graph_embedding = self.lay_pool(node_embedding, data.batch)
        
        output = graph_embedding
        for lay in self.predictor_layers:
            output = lay(output)
        
        return {
            'graph_output': output,
            'graph_embedding': graph_embedding,
        }
        
    def set_encoder(self, model):
        self.lay_embedd = model.lay_embedd
        self.encoder_layers = model.encoder_layers
        

class GATModel(EvidentialMixin, 
               SwagMixin, 
               MeanVarianceMixin, 
               InitializationMixin, 
               AbstractGraphModel
               ):
    """
    Implements a simple GNN architecture based on the GATv2 layer type. 
    
    The model consists of an encoder subnetwork using the GATv2 layers to update the node embeddings
    and a predictor subnetwork that takes the final graph embedding and produces the final output.
    The node embeddings are aggregated into a single graph embedding using a simple sum aggregation.
    
    The model also implements several mixins to provide additional functionality:
    - SwagMixin: Implements the possibility to use the method of "Stochastic Weight Averaging Gaussian" 
      for uncertainty estimation
    - MeanVarianceMixin: Implements the possibility to use the method of "Mean Variance Estimation" for 
      uncertainty estimation. With this method the model is augmented to not only predict the mean of the 
      target variable but also the variance at the same time.
    - InitializationMixin: Implements the possibility to use custom initialization methods for the model's
      weights.
    - EvidentialMixin: Implements the possibility to use the method of "Evidential Deep Learning" for
      uncerainty estimation. With this method, the model is augmented to not only predict the mean of the
      target variable but also the parameters of an inverse gamma distribution which can be used to
      estimate aleatoric and epistemic uncertainty.
    """
    
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 encoder_units: list[int],
                 predictor_units: list[int],
                 hidden_units: int = 128,
                 num_heads: int = 3,
                 **kwargs,
                 ):
        
        AbstractGraphModel.__init__(self)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.encoder_units = encoder_units
        self.predictor_units = predictor_units
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        
        self.hparams.update({
            'node_dim': node_dim,
            'edge_dim': edge_dim,
            'encoder_units': encoder_units,
            'predictor_units': predictor_units,
            'hidden_units': hidden_units,
            'num_heads': num_heads,
        })
        
        # ~ embedding
        # At the beginning we use a single linear layer to create an embedding with a 
        # desired dimensionality out of the input node features.
        prev_features = self.encoder_units[0]
        self.lay_embedd = nn.Linear(
            in_features=node_dim,
            out_features=prev_features,
        )
        
        # ~ graph encoder
        # The encoder subnetwork consists of a number of GATv2 layers that update the node embeddings.
        # The number of units in each layer is defined by the "encoder_units" attribute.
        self.lay_act = nn.LeakyReLU()
        self.encoder_layers = nn.ModuleList([])
        for units in self.encoder_units:
            lay = GATv2Conv(
                in_channels=prev_features,
                out_channels=units,
                edge_dim=edge_dim,
                heads=num_heads,
                concat=False,
                negative_slope=0.2,
            )
            prev_features = units
            self.encoder_layers.append(lay)
        
        # This is the dimension of the graph embedding vector which is the result of the final layer of the 
        # encoder subnetwork. We need to define this attribute as a condition for some of the Mixin's
        self.embedding_dim = prev_features
        
        # The pooling layer can be used to aggregate all of the node embeddings into a single graph embedding.
        self.lay_pool = SumAggregation()
            
        # ~ predictor
        # The predictor subnetwork takes the final graph embedding and produces the final output.
        # The number of units in each layer is defined by the "predictor_units" attribute.
        self.predictor_layers = nn.ModuleList([])
        for i, units in enumerate(self.predictor_units):
            
            if i + 1 >= len(self.predictor_units):
                lay = nn.Linear(prev_features, units)
            else:
                lay = nn.Sequential(
                    nn.Linear(prev_features, units),
                    nn.LeakyReLU(),
                )
            
            prev_features = units
            self.predictor_layers.append(lay)
            
        # ~ implementing mixins
        
        MeanVarianceMixin.__init__(self, **kwargs)
        
        SwagMixin.__init__(self, **kwargs)
        
        EvidentialMixin.__init__(self, **kwargs)
        
        InitializationMixin.__init__(self, **kwargs)
        self.initialize_weights()
            
    def forward(self, data: Data):
        """
        Given the torch geometric ``data`` object that represents the input graph, this method will perform 
        a model forward pass and then return a dictionary with the results of the forward pass. This dictionary 
        will contain at the very least the two entries:
        - graph_output: The final output value of the network which estimates the target property
        - graph_embedding: The final graph embedding vector
        
        :param data: torch geometric data object
        
        :returns: dict
        """
        node_input, edge_input, edge_indices = data.x, data.edge_attr, data.edge_index
        
        node_embedding = self.lay_embedd(node_input)
        for lay in self.encoder_layers:
            node_embedding = lay(node_embedding, edge_indices, edge_input)
            node_embedding = self.lay_act(node_embedding)
            
        graph_embedding = self.lay_pool(node_embedding, data.batch)
        
        output = graph_embedding
        for lay in self.predictor_layers:
            output = lay(output)
        
        return {
            'graph_output': output,
            'graph_embedding': graph_embedding,
        }

class GCNModel(EvidentialMixin, 
               SwagMixin, 
               MeanVarianceMixin, 
               InitializationMixin, 
               AbstractGraphModel):
    """
    Implements a simple GNN architecture based on the GCN layer type.

    The model consists of an encoder subnetwork using the GCN layers to update the node embeddings
    and a predictor subnetwork that takes the final graph embedding and produces the final output.
    The node embeddings are aggregated into a single graph embedding using a simple sum aggregation.

    The model also implements several mixins to provide additional functionality:
    - SwagMixin: Implements the possibility to use the method of "Stochastic Weight Averaging Gaussian" 
      for uncertainty estimation.
    - MeanVarianceMixin: Implements the possibility to use the method of "Mean Variance Estimation" for 
      uncertainty estimation.
    - InitializationMixin: Implements the possibility to use custom initialization methods for the model's
      weights.
    - EvidentialMixin: Implements the possibility to use "Evidential Deep Learning" for uncertainty estimation.
    """
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 encoder_units: list[int],
                 predictor_units: list[int],
                 hidden_units: int = 128,
                 **kwargs):
        AbstractGraphModel.__init__(self)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.encoder_units = encoder_units
        self.predictor_units = predictor_units
        self.hidden_units = hidden_units

        self.hparams.update({
            'node_dim': node_dim,
            'edge_dim': edge_dim,
            'encoder_units': encoder_units,
            'predictor_units': predictor_units,
            'hidden_units': hidden_units,
        })

        # Embedding layer to transform input node features
        prev_features = self.encoder_units[0]
        self.lay_embedd = nn.Linear(
            in_features=node_dim,
            out_features=prev_features,
        )

        # Encoder: Stack of GCNConv layers with activation after each layer
        self.encoder_layers = nn.ModuleList([])
        self.lay_act = nn.LeakyReLU()
        for units in self.encoder_units:
            lay = GCNConv(
                in_channels=prev_features, 
                out_channels=units,
                improved=True,
                add_self_loops=True,
            )
            prev_features = units
            self.encoder_layers.append(lay)

        # This is the dimension of the graph embedding vector which is the result of the final layer of the 
        # encoder subnetwork. We need to define this attribute as a condition for some of the Mixin's
        self.embedding_dim = prev_features
        
        # The pooling layer can be used to aggregate all of the node embeddings into a single graph embedding.
        self.lay_pool = SumAggregation()

        # Predictor network
        self.predictor_layers = nn.ModuleList([])
        for c, units in enumerate(self.predictor_units, start=1):
            if c == len(self.predictor_units):
                lay = nn.Linear(prev_features, units)
            else:
                lay = nn.Sequential(
                    nn.Linear(prev_features, units),
                    nn.LeakyReLU(),
                )
            prev_features = units
            self.predictor_layers.append(lay)

        # Implementing mixins
        MeanVarianceMixin.__init__(self, **kwargs)
        SwagMixin.__init__(self, **kwargs)
        EvidentialMixin.__init__(self, **kwargs)
        InitializationMixin.__init__(self, **kwargs)
        self.initialize_weights()

    def forward(self, data: Data):
        node_input, edge_input, edge_indices = data.x, data.edge_attr, data.edge_index
        
        node_embedding = self.lay_embedd(node_input)
        
        for lay in self.encoder_layers:
            node_embedding = lay(node_embedding, edge_indices)
            node_embedding = self.lay_act(node_embedding)
        
        graph_embedding = self.lay_pool(node_embedding, data.batch)
        
        output = graph_embedding
        for lay in self.predictor_layers:
            output = lay(output)
        
        return {
            'graph_output': output,
            'graph_embedding': graph_embedding,
        }