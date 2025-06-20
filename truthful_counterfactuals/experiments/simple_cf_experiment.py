"""A small experiment which generates counterfactual explanations for a list of
SMILES strings and visualizes them before and after filtering by an
uncertainty threshold."""

import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.util import dynamic_import
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.visualization.molecules import (
    mol_from_smiles,
    visualize_molecular_graph_from_mol,
)
from vgd_counterfactuals.generate.molecules import get_neighborhood

# == EXPERIMENT PARAMETERS ==

MODEL_PATH: str = ''
PROCESSING_PATH: str = ''
SMILES: List[str] = []
UNCERTAINTY_THRESHOLD: float = 1.0
NUM_COUNTERFACTUALS: int = 5
FIG_SIZE: int = 5

__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('load_model', default=True, replace=False)
def load_model(e: Experiment, model_path: str):
    """Load a torch model from the given ``model_path``."""
    import torch

    model = torch.load(model_path, map_location='cpu')
    if hasattr(model, 'eval'):
        model.eval()
    return model


@experiment.hook('predict_graph', default=True, replace=False)
def predict_graph(e: Experiment, model, graphs: List[dict]) -> List[dict]:
    """Return predictions and uncertainties for a batch of graphs."""
    infos = model.forward_graphs(graphs)
    results = []
    for info in infos:
        pred = float(np.squeeze(info.get('graph_output', 0.0)))
        if 'graph_uncertainty' in info:
            unc = float(np.squeeze(info['graph_uncertainty']))
        elif 'graph_std' in info:
            unc = float(np.squeeze(info['graph_std']))
        elif 'graph_log_var' in info:
            unc = float(np.exp(float(np.squeeze(info['graph_log_var'])) / 2))
        else:
            unc = 0.0
        results.append({'graph_prediction': pred, 'graph_uncertainty': unc})
    return results


@experiment
def experiment(e: Experiment):
    """Main experiment callback."""
    e.log('loading model...')
    model = e.apply_hook('load_model', model_path=e.MODEL_PATH)

    e.log('loading processing...')
    module = dynamic_import(e.PROCESSING_PATH)
    processing: MoleculeProcessing = module.processing

    for i, smiles in enumerate(e.SMILES):
        e.log(f'processing {i}: {smiles}')
        graph = processing.process(smiles)
        neighbors_data = get_neighborhood(smiles=smiles, fix_protonation=False)
        neighbor_graphs = [processing.process(d['value']) for d in neighbors_data]

        infos = e.apply_hook('predict_graph', model=model, graphs=[graph] + neighbor_graphs)
        info_org, infos_cf = infos[0], infos[1:]

        graph['graph_prediction'] = info_org['graph_prediction']
        graph['graph_uncertainty'] = info_org['graph_uncertainty']
        for g, info in zip(neighbor_graphs, infos_cf):
            g['graph_prediction'] = info['graph_prediction']
            g['graph_uncertainty'] = info['graph_uncertainty']

        counterfactuals = sorted(
            neighbor_graphs,
            key=lambda g: abs(g['graph_prediction'] - graph['graph_prediction']),
            reverse=True,
        )[: e.NUM_COUNTERFACTUALS]

        fig, axes = plt.subplots(
            ncols=len(counterfactuals) + 1,
            nrows=1,
            figsize=(e.FIG_SIZE * (len(counterfactuals) + 1), e.FIG_SIZE),
            squeeze=False,
        )
        ax = axes[0][0]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#E6F0FA')
        ax.set_title(
            f'Original\nPred: {graph["graph_prediction"]:.2f}\n'
            f'\u03C3: {graph["graph_uncertainty"]:.2f}'
        )
        mol_org = mol_from_smiles(smiles)
        visualize_molecular_graph_from_mol(ax=ax, mol=mol_org, image_width=1000, image_height=1000)

        for j, cf in enumerate(counterfactuals):
            ax = axes[0][j + 1]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('#FFFFFF')
            ax.set_title(
                f'CF {j}\nPred: {cf["graph_prediction"]:.2f}\n'
                f'\u03C3: {cf["graph_uncertainty"]:.2f}'
            )
            mol_cf = mol_from_smiles(cf['graph_repr'])
            visualize_molecular_graph_from_mol(ax=ax, mol=mol_cf, image_width=1000, image_height=1000)

        e.commit_fig(f'example_{i}_pre.pdf', fig)

        counterfactuals_filtered = [
            cf for cf in counterfactuals if cf['graph_uncertainty'] <= e.UNCERTAINTY_THRESHOLD
        ]
        if counterfactuals_filtered:
            fig, axes = plt.subplots(
                ncols=len(counterfactuals_filtered) + 1,
                nrows=1,
                figsize=(e.FIG_SIZE * (len(counterfactuals_filtered) + 1), e.FIG_SIZE),
                squeeze=False,
            )
            ax = axes[0][0]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('#E6F0FA')
            ax.set_title(
                f'Original\nPred: {graph["graph_prediction"]:.2f}\n'
                f'\u03C3: {graph["graph_uncertainty"]:.2f}'
            )
            visualize_molecular_graph_from_mol(
                ax=ax,
                mol=mol_org,
                image_width=1000,
                image_height=1000,
            )
            for j, cf in enumerate(counterfactuals_filtered):
                ax = axes[0][j + 1]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor('#FFFFFF')
                ax.set_title(
                    f'CF {j}\nPred: {cf["graph_prediction"]:.2f}\n'
                    f'\u03C3: {cf["graph_uncertainty"]:.2f}'
                )
                mol_cf = mol_from_smiles(cf['graph_repr'])
                visualize_molecular_graph_from_mol(
                    ax=ax,
                    mol=mol_cf,
                    image_width=1000,
                    image_height=1000,
                )

            e.commit_fig(f'example_{i}_post.pdf', fig)


experiment.run_if_main()
