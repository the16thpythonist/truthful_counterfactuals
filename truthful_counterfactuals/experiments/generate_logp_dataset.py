import os
import time
import csv
import shutil
import datetime

import numpy as np
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.util import dynamic_import
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.data import VisualGraphDatasetWriter

from truthful_counterfactuals.utils import EXPERIMENTS_PATH


# == SOURCE PARAMETERS ==

PROCESSING_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'process.py')
CSV_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'molecules.csv')
SMILES_COLUMN: str = 'smiles'

__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('calculate_target', default=True, replace=False)
def calculate_target(e: Experiment, data: dict) -> list:
    smiles = data[e.SMILES_COLUMN]
    mol = Chem.MolFromSmiles(smiles)
    
    # The target value that we'll be using is crippens logp value
    target = Chem.Crippen.MolLogP(mol)
    
    return [float(target)]


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    
    # ~ loading the sources
    
    e.log('loading the processing module...')
    module = dynamic_import(e.PROCESSING_PATH)
    processing: MoleculeProcessing = module.processing
    e.log(f'loaded processing: {processing}')

    e.log('loading the source dataset...')
    data_list = []
    with open(CSV_PATH, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data_list.append(row)
    
    e.log(f'loaded the dataset with {len(data_list)} elements')
            
    # ~ creating the target values
    
    e.log('calculating target values...')
    for data in data_list:
        data['target'] = e.apply_hook(
            'calculate_target',
            data=data,
        )
        
    # ~ exporting as visual graph dataset
    
    e.log('creating the visual graph dataset...')
    dataset_path = os.path.join(e.path, 'dataset')
    os.mkdir(dataset_path)
    
    writer = VisualGraphDatasetWriter(dataset_path)
    
    time_start = time.time()
    for index, data in enumerate(data_list):
        
        processing.create(
            value=data[e.SMILES_COLUMN],
            index=index,
            additional_graph_data={'graph_labels': data['target']},
            additional_metadata={'target': data['target']},
            writer=writer,
        )
        
        if index % 1000 == 0:
            time_elapsed = time.time() - time_start
            time_remaining = time_elapsed / (index + 1) * (len(data_list) - index - 1)
            eta = datetime.datetime.now() + datetime.timedelta(seconds=time_remaining)
            e.log(f' * {index}/{len(data_list)} processed''
                  f' - smiles: {data["smiles"]}'
                  f' - target: {data["target"]}')
        
    # In the end we also want to copy the processing module to the actual dataset folder so that 
    # this can be used independently in the future.
    shutil.copy(e.PROCESSING_PATH, dataset_path)


experiment.run_if_main()