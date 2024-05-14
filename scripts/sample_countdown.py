import os, sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.getcwd())

from config.eval.countdown import get_config as get_eval_config
import lib.utils.bookkeeping as bookkeeping
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import lib.utils.utils as utils
import lib.models.models as models
import lib.models.model_utils as model_utils
import lib.datasets.datasets as datasets
import lib.datasets.dataset_utils as dataset_utils
import lib.sampling.sampling as sampling
import lib.sampling.sampling_utils as sampling_utils

# Create the parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--sampler', type=str, default="birthdeath", help='')
parser.add_argument('--num-steps', type=int, default=500, help='')
parser.add_argument('--corrector-steps', type=int, default=2, help='')
parser.add_argument('--entry-time', type=float, default=0.9, help='')
parser.add_argument('--corrector-step-size', type=float, default=.5, help='')

# Parse the arguments
args = parser.parse_args()

eval_cfg = get_eval_config()
train_cfg = bookkeeping.load_ml_collections(Path(eval_cfg.train_config_path))

for item in eval_cfg.train_config_overrides:
    utils.set_in_nested_dict(train_cfg, item[0], item[1])

print("Sampler: {}".format(args.sampler))
print("Number of corrector steps: {}".format(args.corrector_steps))
print("Corrector entry time: {}".format(args.entry_time))
print("Corrector step size multiplier: {}".format(args.corrector_step_size))

# Override default configs
eval_cfg.sampler.name = 'PCTauLeapingAbsorbingInformed'
eval_cfg.sampler.num_steps = args.num_steps
eval_cfg.sampler.num_corrector_steps = args.corrector_steps
eval_cfg.sampler.corrector_entry_time = args.entry_time
eval_cfg.sampler.corrector_step_size_multiplier = args.corrector_step_size
eval_cfg.sampler.balancing_function = args.sampler

S = train_cfg.data.S 
device = torch.device("cuda")

model = model_utils.create_model(train_cfg, device)

loaded_state = torch.load(Path(eval_cfg.checkpoint_path),
    map_location=device)

modified_model_state = utils.remove_module_from_keys(loaded_state['model'])
model.load_state_dict(modified_model_state)

model.eval()

dataset = dataset_utils.get_dataset(eval_cfg, device)
data_flat = dataset.data.flatten().cpu().numpy()

def count_mistakes(samples):
    """
    sample: (N, L, S)
    """
    num_positions = samples.shape[0] * samples.shape[1]
    return (np.sum(((samples[:,:-1] - samples[:,1:]) != 1) * (samples[:,:-1] != 0))) / num_positions

def get_dist(seq, S):
    L = seq.shape[0]
    one_hot = np.zeros((L, S))
    seq = np.array(seq, dtype=int)
    one_hot[np.arange(L), seq] = 1
    return np.sum(one_hot, axis=0) / L

def hellinger(seq1, seq2, S):
    d1, d2 = get_dist(seq1, S), get_dist(seq2, S)
    return np.sqrt(.5 * np.sum((d1 ** .5 - d2 ** .5) ** 2))

sample_size = 1000
batch_size = 100
num_repeats = 5

sampler = sampling_utils.get_sampler(eval_cfg)

h_dists = []
error_rates = []
for _ in range(num_repeats):
    all_samples = []
    for start in range(0, sample_size, batch_size):
        print(start)
        end = min(start + batch_size, sample_size)
        size = end - start
        
        samples, _ = sampler.sample(model, size, 1)

        all_samples.append(samples)
    all_samples = np.concatenate(all_samples, axis=0)
    error_rate = count_mistakes(all_samples)
    h_dist = hellinger(data_flat, all_samples.flatten(), S)
            
    print("Hellinger distance", h_dist)
    print("Error rate", error_rate)