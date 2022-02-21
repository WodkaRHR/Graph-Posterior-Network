from gpn.data import DatasetManager, get_ood_split_evasion
from gpn.data import InMemoryDatasetProvider, OODInMemoryDatasetProvider, FileExperimentDatasetProvider
from gpn.data import OODIsolatedInMemoryDatasetProvider
from gpn.utils import DataConfiguration, FileDataConfiguration
import torch_geometric as tg
import os.path as osp
import pickle
import torch

from gpn.utils.config import FileDataConfiguration

def set_num_left_out(data_cfg: DataConfiguration):
    """utility function setting the number-of-left-out classes for LOC experiments for each dataset accordingly

    Args:
        data_cfg (DataConfiguration): original data configuration

    Raises:
        ValueError: raised if unsupported dataset found
    """

    if data_cfg.dataset in ('Cora', 'CoraML'):
        data_cfg.set_values(ood_num_left_out_classes=3)

    elif data_cfg.dataset == 'CoraFull':
        data_cfg.set_values(ood_num_left_out_classes=30)

    elif 'CiteSeer' in data_cfg.dataset:
        data_cfg.set_values(ood_num_left_out_classes=2)

    elif 'PubMed' in data_cfg.dataset:
        data_cfg.set_values(ood_num_left_out_classes=1)

    elif data_cfg.dataset == 'AmazonPhotos':
        data_cfg.set_values(ood_num_left_out_classes=3)

    elif data_cfg.dataset == 'AmazonComputers':
        data_cfg.set_values(ood_num_left_out_classes=5)

    elif data_cfg.dataset == 'ogbn-arxiv':
        data_cfg.set_values(ood_num_left_out_classes=15)

    elif data_cfg.dataset == 'CoauthorPhysics':
        data_cfg.set_values(ood_num_left_out_classes=2)

    elif data_cfg.dataset == 'CoauthorCS':
        data_cfg.set_values(ood_num_left_out_classes=4)

    else:
        raise ValueError(f'Dataset {data_cfg.dataset} not supported!')


class FileExperimentDataset:
    """ wrapper for dataset to be used in an experiment from a file """

    def __init__(self, data_cfg: FileDataConfiguration):
        self.data_cfg = data_cfg

        with open(osp.join(data_cfg.directory, f'split-{data_cfg.split_no - 1}.pkl'), 'rb') as f:
            storage = pickle.load(f)
        
        developement_data = storage['data']['train']
        developement_data.train_mask = storage['data']['train'].mask
        developement_data.val_mask = storage['data']['val'].mask
        developement_data.test_mask = storage['data']['test'].mask
        developement_dataset = FileExperimentDatasetProvider(developement_data)
        self.train_dataset = developement_dataset
        self.val_dataset = developement_dataset
        self.train_val_dataset = developement_dataset

        ood_data = storage['data'][data_cfg.ood_val_dataset] 

        def developement_mask_to_ood_mask(dev_mask):
            # Translates a mask on the developement graph to a mask on the ood graph
            vertices = set(v for v, idx in storage['data']['train'].vertex_to_idx.items() if dev_mask[idx])
            ood_mask = torch.zeros_like(storage['data'][data_cfg.ood_val_dataset].mask).bool()
            for v in vertices:
                ood_mask[storage['data'][data_cfg.ood_val_dataset].vertex_to_idx[v]] = True
            return ood_mask

        ood_data.train_mask = developement_mask_to_ood_mask(storage['data']['train'].mask)
        ood_data.val_mask = developement_mask_to_ood_mask(storage['data']['val'].mask)
        ood_data.test_mask = developement_mask_to_ood_mask(storage['data']['test'].mask)
        ood_data.ood_mask = ~storage['data'][data_cfg.ood_val_dataset].is_in_distribution
        ood_data.id_mask = storage['data'][data_cfg.ood_val_dataset].is_in_distribution
        ood_data.ood_val_mask = storage['data'][data_cfg.ood_val_dataset].mask & (~storage['data'][data_cfg.ood_val_dataset].is_in_distribution) & storage['data'][data_cfg.ood_val_dataset]['auroc_mask']
        ood_data.ood_test_mask = storage['data'][data_cfg.ood_test_dataset].mask & (~storage['data'][data_cfg.ood_val_dataset].is_in_distribution) & storage['data'][data_cfg.ood_test_dataset]['auroc_mask']
        ood_data.id_val_mask = storage['data'][data_cfg.ood_val_dataset].mask & (storage['data'][data_cfg.ood_val_dataset].is_in_distribution) & storage['data'][data_cfg.ood_val_dataset]['auroc_mask']
        ood_data.id_test_mask = storage['data'][data_cfg.ood_test_dataset].mask & (storage['data'][data_cfg.ood_val_dataset].is_in_distribution) & storage['data'][data_cfg.ood_test_dataset]['auroc_mask']
        
        ood_dataset = FileExperimentDatasetProvider(ood_data)
        self.ood_dataset = ood_dataset
    
        self.splits = ('train', 'test', 'val')

        # finally reset number of classes
        self.num_classes = len(torch.unique(developement_data.y[developement_data.train_mask]))
        self.dim_features = developement_data.x.size(1)

        # if nothing further specified: warmup/finetuning on training dataset
        self.warmup_dataset = self.train_dataset
        self.finetune_dataset = self.train_dataset

        self.train_loader = None
        self.train_val_loader = None
        self.val_loader = None
        self.ood_loader = None
        self.warmup_loader = None
        self.finetune_loader = None

        self.setup_loader()

    def setup_loader(self):
        self.train_loader = self.train_dataset.loader()
        self.train_val_loader = self.train_val_dataset.loader()
        self.val_loader = self.val_dataset.loader()

        if self.ood_dataset is not None:
            self.ood_loader = self.ood_dataset.loader()
        else:
            self.ood_loader = None

        if self.warmup_dataset is not None:
            self.warmup_loader = self.warmup_dataset.loader()
        else:
            self.warmup_loader = None

        if self.finetune_dataset is not None:
            self.finetune_loader = self.finetune_dataset.loader()
        else:
            self.finetune_loader = None

class ExperimentDataset:
    """wrapper for dataset to be used in an experiment

    Sets up the dataset as specified for all different kinds of experiments, e.g. OOD experiments.
    """

    def __init__(self, data_cfg: DataConfiguration, to_sparse: bool = False):
        self.data_cfg = data_cfg

        for _ in range(data_cfg.split_no):
            dataset = DatasetManager(**data_cfg.to_dict())

        default_dataset = InMemoryDatasetProvider(dataset)

        self.dim_features = default_dataset.num_features
        self.num_classes = default_dataset.num_classes

        self.train_dataset = default_dataset
        self.train_val_dataset = default_dataset
        self.val_dataset = default_dataset
        self.ood_dataset = None

        self.to_sparse = to_sparse

        self.splits = ('train', 'test', 'val', 'all')

        if data_cfg.ood_flag:
            if data_cfg.ood_setting == 'evasion':
                self._setup_evasion()

            elif data_cfg.ood_setting == 'poisoning':
                self.splits = ('train', 'test', 'val')
                self._setup_poisoning()

            else:
                raise ValueError

        else:
            if to_sparse:
                self.train_dataset.to_sparse()

        # finally reset number of classes
        self.num_classes = self.train_dataset.num_classes

        # if nothing further specified: warmup/finetuning on training dataset
        self.warmup_dataset = self.train_dataset
        self.finetune_dataset = self.train_dataset

        self.train_loader = None
        self.train_val_loader = None
        self.val_loader = None
        self.ood_loader = None
        self.warmup_loader = None
        self.finetune_loader = None

        self.setup_loader()

    def setup_loader(self):
        self.train_loader = self.train_dataset.loader()
        self.train_val_loader = self.train_val_dataset.loader()
        self.val_loader = self.val_dataset.loader()

        if self.ood_dataset is not None:
            self.ood_loader = self.ood_dataset.loader()
        else:
            self.ood_loader = None

        if self.warmup_dataset is not None:
            self.warmup_loader = self.warmup_dataset.loader()
        else:
            self.warmup_loader = None

        if self.finetune_dataset is not None:
            self.finetune_loader = self.finetune_dataset.loader()
        else:
            self.finetune_loader = None

    def _setup_evasion(self):
        # in evasion setting, also allow perturbations
        # of the set of training nodes
        # budget: val_dataset and ood_dataset are the same
        # isolated: val dataset is not perturbed, ood set is perturbed
        if self.data_cfg.ood_dataset_type == 'budget':
            self.ood_dataset = OODInMemoryDatasetProvider(self.val_dataset)
            self.ood_dataset.perturb_dataset(**{**self.data_cfg.to_dict(), 'perturb_train_indices': True})

            self.val_dataset = self.ood_dataset
            if self.to_sparse:
                self.ood_dataset.to_sparse()

        elif self.data_cfg.ood_dataset_type == 'isolated':
            self.ood_dataset = OODIsolatedInMemoryDatasetProvider(
                self.val_dataset, self.data_cfg.ood_type, **self.data_cfg.to_dict())

            if self.to_sparse:
                self.val_dataset.to_sparse()

        else:
            raise ValueError

    def _setup_poisoning(self):
        # train dataest is perturbed (new train_loader)
        # val_loader and ood_loader are based on perturbed val_dataset
        if self.data_cfg.ood_type == 'leave_out_classes':
            set_num_left_out(self.data_cfg)

            self.train_dataset = OODInMemoryDatasetProvider(self.train_dataset)
            self.train_dataset.perturb_dataset(**{**self.data_cfg.to_dict(), 'perturb_train_indices': True})


            if self.to_sparse:
                self.train_dataset.to_sparse()
            self.train_val_dataset = self.train_dataset
            self.ood_dataset = self.train_dataset
            self.val_dataset = self.train_dataset

        elif self.data_cfg.ood_type == 'leave_out_classes_evasion':
            assert len(self.train_dataset) == 1
            set_num_left_out(self.data_cfg)

            id_data, ood_data, num_classes = get_ood_split_evasion(
                self.train_dataset[0],
                num_classes=self.train_dataset.num_classes,
                perturb_train_indices=True,
                **self.data_cfg.to_dict()
            )

            self.train_dataset.data_list = [id_data]
            self.train_dataset.set_num_classes(num_classes)
            self.train_val_dataset = self.train_dataset

            if self.to_sparse:
                self.train_dataset.to_sparse()

            self.val_dataset = self.train_dataset.clone(shallow=True)
            self.val_dataset.data_list = [ood_data]
            self.val_dataset.set_num_classes(num_classes)
            self.ood_dataset = self.val_dataset

            if self.to_sparse:
                self.val_dataset.to_sparse()

        elif self.data_cfg.ood_dataset_type == 'budget':
            self.train_dataset = OODInMemoryDatasetProvider(self.train_dataset)
            self.train_dataset.perturb_dataset(**{**self.data_cfg.to_dict(), 'perturb_train_indices': False})

            self.train_val_dataset = self.train_dataset
            self.ood_dataset = self.train_val_dataset
            self.val_dataset = self.train_val_dataset

            if self.to_sparse:
                self.train_dataset.to_sparse()

        else:
            raise ValueError
