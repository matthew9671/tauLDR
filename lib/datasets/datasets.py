import torch
from torch.utils.data import Dataset
from torch.distributions import Categorical
from . import dataset_utils
import numpy as np
import torchvision.datasets
import torchvision.transforms
import os

import pandas as pd

@dataset_utils.register_dataset
class DiscreteCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, cfg, device):
        super().__init__(root=cfg.data.root, train=cfg.data.train,
            download=cfg.data.download)

        self.data = torch.from_numpy(self.data)
        self.data = self.data.transpose(1,3)
        self.data = self.data.transpose(2,3)

        self.targets = torch.from_numpy(np.array(self.targets))

        # Put both data and targets on GPU in advance
        self.data = self.data.to(device).view(-1, 3, 32, 32)

        self.random_flips = cfg.data.random_flips
        if self.random_flips:
            self.flip = torchvision.transforms.RandomHorizontalFlip()


    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'CIFAR10', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'CIFAR10', 'processed')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.random_flips:
            img = self.flip(img)

        return img


@dataset_utils.register_dataset
class LakhPianoroll(Dataset):
    def __init__(self, cfg, device):
        S = cfg.data.S
        L = cfg.data.shape[0]
        np_data = np.load(cfg.data.path) # (N, L) in range [0, S)

        self.data = torch.from_numpy(np_data).to(device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        return self.data[index]
    
@dataset_utils.register_dataset
class Countdown(Dataset):
    def __init__(self, cfg, device):
        S = cfg.data.S - 1
        L = cfg.data.shape[0]
        N = cfg.data.data_size
        total_len = N * L
        data = torch.zeros((total_len,))
        cat = Categorical(torch.ones((S,))/S)
        starts = cat.sample((total_len,))
        curr = 0
        while curr < total_len:
            x = starts[curr]
            l = min(total_len-1, curr+x) - curr
            data[curr:curr+l] = torch.arange(x, x-l, -1)
            curr += x + 1

        self.data = data.view(N, L).to(device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        return self.data[index]
    
@dataset_utils.register_dataset
class DiscreteGenes(Dataset):
    def __init__(self, cfg, device):

        if cfg.data.use_absorbing:
            S = cfg.data.S - 1
        else:
            S = cfg.data.S

        full_data = torch.load(cfg.data.path).double()
        centered_loggen = full_data.log10() - full_data.log10().mean(1).unsqueeze(-1)
        expr_genes = centered_loggen[(centered_loggen.std(1) > .3),:]

        self.data_min = expr_genes.min().numpy()
        self.data_max = expr_genes.max().numpy()

        self.S = S
        self.data = dataset_utils.discretize_data(expr_genes, S).to(device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        return self.data[index]
    
    def rescale(self, d):
        """
        Rescale discretized data back to original range.
        """
        return d / self.S * (self.data_max - self.data_min) + self.data_min
    
@dataset_utils.register_dataset
class DiscreteAEMET(Dataset):
    def __init__(self, cfg, device):

        if cfg.data.use_absorbing:
            S = cfg.data.S - 1
        else:
            S = cfg.data.S

        data_raw = pd.read_csv(cfg.data.path, index_col=0)
        data = torch.tensor(data_raw.values).squeeze()

        min_data = torch.min(data)
        max_data = torch.max(data)
        rescaled_data = 6 * (data - min_data) / (max_data - min_data) - 3
        n_repeat = 50
        rescaled_data_repeated = rescaled_data.repeat(n_repeat, 1)

        self.data_min = -3.0
        self.data_max = 3.0

        self.S = S
        self.data = dataset_utils.discretize_data(rescaled_data_repeated, S).to(device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        return self.data[index]
    
    def rescale(self, d):
        """
        Rescale discretized data back to original range.
        """
        return d / self.S * (self.data_max - self.data_min) + self.data_min