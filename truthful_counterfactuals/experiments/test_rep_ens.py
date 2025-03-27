import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

__DEBUG__ = True

# == MLP MODEL ==
class SimpleMLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=256, out_dim=1, act_fn = lambda: nn.GELU()):
        super().__init__()
        
        latent_dim = 256
        self.net_base = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        self.net_avg = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, out_dim),
        )
        
        self.net_var = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, out_dim),
        )
            
    def get_encoder(self) -> nn.Module:
        return self.net_base
            
    def set_encoder(self, module: nn.Module):
        self.net_base = module

    def forward(self, x: torch.Tensor, detach=False) -> torch.Tensor:
        emb = self.net_base(x)
        emb = F.normalize(emb, dim=1, p=2)
            
        return {
            'out': self.net_avg(emb), 
            'var': self.net_var(emb),
            'emb': emb
        }

# == REPULSIVE ENSEMBLE FOR MLP ==
class RepulsiveEnsemble(pl.LightningModule):
    
    def __init__(self, 
                 models, 
                 learning_rate=1e-2, 
                 repulsive_factor=0.01
                 ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.learning_rate = learning_rate
        self.repulsive_factor = repulsive_factor
        self.criterion = nn.MSELoss()
        self.num_models = len(models)
        
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: 
        return self.loss_pred(x, y)
    
    def loss_pred(self, model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p, v = model(x)
        return (p - y).pow(2).mean()
    
    def loss_mve(self, model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        info = self(x)
        p = info['out']
        v = info['var']
        loss_pred = ((p - y).pow(2) / (2 * v**2)).mean()
        loss_nll = 0.5 * torch.log(v**2)
        return loss_pred + loss_nll

    def forward(self, x):
        preds = []
        sigs = []
        for model in self.models:
            info = model(x)
            p = info['out']
            v = info['var']
            preds.append(p)
            sigs.append(v)
            
        stack = torch.stack(preds, dim=0)
        sig_al = torch.stack(sigs, dim=0).mean(dim=0)
        sig_ep = torch.stack(preds, dim=0).std(dim=0)
        
        return stack.mean(dim=0), sig_ep, sig_al

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.size(0)
        loss = 0.0
        
        # Determine the current epoch
        current_epoch = self.current_epoch
        total_epochs = self.trainer.max_epochs

        if current_epoch < total_epochs / 2 or True:
            
            # standard prediction loss for the members
            x.requires_grad = True
            out_preds = []
            out_grads = []
            out_true = y
            loss_pred = 0.0
            for model in self.models:
                info = model(x)
                p = info['out']
                v = info['var']
                out_preds.append(p)
                
                emb = info['emb']
                # Compute the first principal component of the embedding vectors
                emb_centered = emb - emb.mean(dim=0, keepdim=True)
                u, s, v = torch.svd(emb_centered)
                pc = v[:, 0:1]
                
                inp = x
                inp = emb
                g = torch.autograd.grad(p.sum(), inp, create_graph=True)[0]
                #print(g.shape, v.shape, pc.shape)
                #proj = (g * pc.detach()).sum(dim=1)
                proj = torch.matmul(g, pc)
                #print(g.shape, v.shape, proj.shape)
                out_grads.append(proj)
                
                loss_pred += (p - y).pow(2).mean()
            
            loss += loss_pred / len(self.models)
            # out_preds_stacked = torch.stack(out_preds, dim=0)
            # loss += 0.01 * torch.std(out_preds_stacked, dim=0).mean()
            self.log('loss_pred', loss_pred, prog_bar=True, on_epoch=True)

            # ~ repulsive loss
            # In addition to the prediction loss, we also want to compute a repulsive loss term
            # which promotes the models to have a pairwise different output function.
            loss_rep = 0.0
            # This is the scalar average error over the current mini batch
            # avg_error: (1, )
            avg_error = torch.stack([(out_pred - out_true).abs() for out_pred in out_preds])
            avg_error = avg_error.mean()
            self.log('avg_error', avg_error, prog_bar=False, on_epoch=True, batch_size=batch_size)
            # This is a term proportional to the disagreement between all the models in the ensemble 
            # for each of the elements in the batch.
            # mdl_disagree: (batch_size, )
            # mdl_disagree = torch.stack(out_preds, dim=0)
            # mdl_disagree = torch.std(mdl_disagree, dim=0)
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
            # This is done by computing the squared difference of the gradients of the two models
            # and then multiplying that with the alpha value which is a measure of how much the models
            # disagree on that specific element. We want to maximize the disagreement between the models 
            # preferably on those elements which they already disagree on.
            num = 1
            for _ in range(num):
                i, j = random.sample(range(self.num_models), 2)
                dist = (out_grads[i] - out_grads[j]).abs().mean(dim=-1)
                #print(dist.shape)
                #dist = torch.clamp_min(100 - dist, 0)
                loss_rep_ = (alpha.unsqueeze(-1).detach().clone() * (dist - 10.0).pow(2))
                #loss_rep_ = dist
                loss_rep_ = loss_rep_.mean()
                
                # g_i_norm = F.normalize(out_grads[i], dim=1)
                # g_j_norm = F.normalize(out_grads[j], dim=1)
                # loss_rep_ = (alpha.unsqueeze(-1) * (g_i_norm * g_j_norm).sum(dim=1).abs()).mean()
                
                #loss_rep_ = torch.clamp(loss_rep_, -1)
                loss_rep += (1 / num) * loss_rep_
            
            self.log('loss_rep', loss_rep, prog_bar=True, on_epoch=True, batch_size=batch_size)
            loss += self.repulsive_factor * loss_rep
        
        else:
            
            loss = 0.0
            loss_pred = 0.0
            for model in self.models:
                p, v = model(x, detach=True)
                
                loss_pred += ((p.detach() - y).pow(2)/(v)).mean()
                loss_pred += (torch.log(v)).mean()
                #loss_pred += 0.1 * v.abs().mean()
                
            loss += loss_pred / len(self.models)
            
            self.log('loss_mve', loss_pred, prog_bar=False, on_epoch=True)
            
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# == DATA GENERATION ==
def generate_toy_data(n=50):
    # Sample data from a function only in specific input regions
    xvals = []
    yvals = []
    func = lambda x: np.sin(x) + 0.2*x
    for region in [
        (-6.0, -5),
        (-3.8, -3),
        (-2.0, -1.5),
        (-0.4, 0.1),
        (0.8, 1.2),
        (2.8, 3.2),
        (3.8, 4.5),
        (5.4, 6.0)
        ]:
        x_region = np.random.uniform(*region, size=n)
        y_region = func(x_region)
        if region == (-0.9, -0.85):
            pass 
            #y_region += np.random.normal(0, 0.01, size=n)  # Add Gaussian noise to the first region
        xvals += list(x_region)
        yvals += list(y_region)
    xvals = np.array(xvals).reshape(-1, 1).astype(np.float32)
    yvals = np.array(yvals).reshape(-1, 1).astype(np.float32)
    return xvals, yvals

# == EXPERIMENT ==
experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment
def experiment(e: Experiment):
    # Generate data
    x, y = generate_toy_data()
    dataset = list(zip(torch.from_numpy(x), torch.from_numpy(y)))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Create MLP ensemble
    models = []
    encoder = None
    for _ in range(5):
        model = SimpleMLP()
        if encoder is None:
            encoder = model.get_encoder()
        else:
            model.set_encoder(encoder)
    
        models.append(model)
    
    ensemble = RepulsiveEnsemble(models=models, learning_rate=1e-4, repulsive_factor=0e-4)

    class TrackingCallback(pl.Callback):
        
        def on_train_epoch_end(self, trainer, pl_module):
            super().on_train_epoch_end(trainer, pl_module)
            
            # Iterate through all logged values
            for key, value in trainer.callback_metrics.items():
                if value is not None and 'epoch' in key:
                    # Track each value using the experiment's track method
                    e.track(key, value.item())

    # Train
    trainer = pl.Trainer(max_epochs=200, callbacks=[TrackingCallback()])
    trainer.fit(ensemble, train_dataloaders=train_loader)

    # Plot results
    ensemble.eval()
    for model in ensemble.models:
        model.eval()
    x_plot = np.linspace(-7, 7, 500, dtype=np.float32)
    x_t = torch.from_numpy(x_plot).unsqueeze(-1)
    with torch.no_grad():
        mean, std_ep, std_al = ensemble(x_t)
        individual_preds = [model(x_t)['out'].squeeze().numpy() for model in ensemble.models]
    mean_np, std_ep_np, std_al_np = mean.squeeze().numpy(), std_ep.squeeze().numpy(), std_al.squeeze().numpy()

    fig, ax = plt.subplots()
    ax.plot(x_plot, np.sin(x_plot) + 0.2*x_plot, label='True function', color='green')
    ax.scatter(x, y, label='Train Data', color='blue')
    ax.plot(x_plot, mean_np, label='Ensemble mean', color='red')
    ax.fill_between(x_plot, mean_np - 2*std_ep_np, mean_np + 2*std_ep_np, color='red', alpha=0.2, label='±2 std epistemic')
    #ax.fill_between(x_plot, mean_np - 2*std_al_np, mean_np + 2*std_al_np, color='orange', alpha=0.2, label='±2 std aleatoric')

    # Plot individual curves of ensemble members
    for i, pred in enumerate(individual_preds):
        ax.plot(x_plot, pred, color='red', linestyle='-', alpha=0.1)

    ax.legend()
    e.commit_fig('result.png', fig)
    plt.show()

experiment.run_if_main()
