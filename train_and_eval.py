import os
from typing import OrderedDict
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import logging
import torch
import pandas as pd
import numpy as np

from collections import OrderedDict
from sacred import Experiment
from gpn.utils import RunConfiguration, DataConfiguration, FileDataConfiguration
from gpn.utils import ModelConfiguration, TrainingConfiguration
from gpn.experiments import MultipleRunExperiment
import seml

import pickle

ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.config
def config():
    #pylint: disable=missing-function-docstring
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run_experiment(run: dict, data: dict, model: dict, training: dict, file_dataset : bool=False) -> dict:
    """main function to run experiment with sacred support

    Args:
        run (dict): configuration parameters of the job to run
        data (dict): configuration parameters of the data
        model (dict): configuration parameters of the model
        training (dict): configuration paramterers of the training
        file_dataset (bool): if a file-based dataset will be loaded (from the MastersThesis project)

    Returns:
        dict: numerical results of the evaluation metrics for different splits
    """

    run_cfg = RunConfiguration(**run)
    if file_dataset:
        data_cfg = FileDataConfiguration(**data)
    else:
        data_cfg = DataConfiguration(**data)
    model_cfg = ModelConfiguration(**model)
    train_cfg = TrainingConfiguration(**training)

    if torch.cuda.device_count() <= 0:
        run_cfg.set_values(gpu=False)

    logging.info('Received the following configuration:')
    logging.info('RUN')
    logging.info(run_cfg.to_dict())
    logging.info('-----------------------------------------')
    logging.info('DATA')
    logging.info(data_cfg.to_dict())
    logging.info('-----------------------------------------')
    logging.info('MODEL')
    logging.info(model_cfg.to_dict())
    logging.info('-----------------------------------------')
    logging.info('TRAINING')
    logging.info(train_cfg.to_dict())
    logging.info('-----------------------------------------')

    experiment = MultipleRunExperiment(run_cfg, data_cfg, model_cfg, train_cfg, ex=ex)
    
    results = experiment.run()

    with open(os.path.join(run_cfg.experiment_directory, run_cfg.experiment_name, 'results.pkl'), 'wb+') as f:
        pickle.dump(results, f)

    metrics = [m[4:] for m in results.keys() if m.startswith('val_')]
    result_values = {'val': [], 'test': []}
    
    for s in ('val', 'test'):
        for m in metrics:
            key = f'{s}_{m}'
            if key in results:
                val = results[key]
                if isinstance(val, list):
                    val = np.mean(val)
                result_values[s].append(val)
            else:
                result_values[s].append(None)

    print()
    df = pd.DataFrame(data=result_values, index=metrics)
    print(df.to_markdown())
    print()

    return results
