import torch

_DATASETS = {}

def register_dataset(cls):
    name = cls.__name__
    if name in _DATASETS:
        raise ValueError(f'{name} is already registered!')
    _DATASETS[name] = cls
    return cls

def get_dataset(cfg, device):
    return _DATASETS[cfg.data.name](cfg, device)

def discretize_data(tensor, N):
    # Normalize data to [0, 1]
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    
    # Discretize to N states
    discretized_tensor = torch.floor(normalized_tensor * N)
    
    # Ensure values are within [0, N-1]
    discretized_tensor = torch.clamp(discretized_tensor, 0, N-1)
    
    return discretized_tensor.long()