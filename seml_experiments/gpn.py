from re import M
from collections.abc import Mapping
import yaml
from copy import deepcopy
import os
import os.path as osp


def flatten_configuration(cfg, prefix=[]):
    result = {}
    for k, v in cfg.items():
        if isinstance(v, Mapping):
            result |= flatten_configuration(v, prefix + [k])
        else:
            result['.'.join(prefix + [k])] = v
    return result

if __name__ == '__main__':
    dataset_dir = osp.join(osp.expanduser('~'), 'MastersThesis', '.exported_datasets')
    datasets = os.listdir(dataset_dir)
    output_base = osp.join(osp.dirname(__file__), 'gpn')

    with open(osp.join(osp.dirname(__file__), 'gpn_base.yaml')) as f:
        gpn_base = flatten_configuration(yaml.safe_load(f))
    

    output_yaml = {
        'seml' : {
            'executable' : 'train_and_eval.py',
            'name' : f'gpn_baseline',
            'output_dir' : '/nfs/students/fuchsgru/seml_output/gpn',
            'project_root_dir' : '../..'
        },
        'slurm' : {
            'experiments_per_job' : 2,
            'sbatch_options' : {
                'gres' : 'gpu:1',
                'mem' : '64GB',
                'cpus-per-task' : 4,
                'time' : '0-72:00'
            }
        },
        'fixed' : deepcopy(gpn_base),
    }
    os.makedirs(output_base, exist_ok=True)

    for dataset in datasets:
        experiments = {}

        for setting in ('hybrid', 'transductive'):
            for name, dataset_suffix, ood_dir in (
                ('loc', '', 'left-out-classes' ),
                ('ber', '-ber', 'perturbations'),
                ('normal', '-normal', 'perturbations')
            ):
                splits_dir = osp.join(dataset_dir, dataset, f'{setting}-{ood_dir}')
                num_splits = (len(os.listdir(splits_dir)))
                assert num_splits in (1, 5), f'Do we really have the right number of splits {num_splits}'
                experiments[f'{dataset}-{setting}-{name}'] = {
                    'fixed' : {
                        'run.num_splits' : num_splits,
                        'run.experiment_name' : f'{setting}-{name}',
                        'run.experiment_directory' : osp.join('saved_experiments', 'gpn', dataset),
                        'data.directory' : splits_dir,
                        'data.ood_val_dataset' : f'ood-val{dataset_suffix}',
                        'data.ood_test_dataset' : f'ood-test{dataset_suffix}',
                    },                    
                }

        with open(osp.join(output_base, f'{dataset}.yaml'), 'w+') as f:
            yaml.safe_dump(deepcopy(experiments | output_yaml), f)

