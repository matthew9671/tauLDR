import os, sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.getcwd())
# !!!
from config.eval.piano_hollow import get_config as get_eval_config
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

print("Using model from " + eval_cfg.checkpoint_path)
print("Sampler: {}".format(args.sampler))
print("Number of corrector steps: {}".format(args.corrector_steps))
print("Corrector entry time: {}".format(args.entry_time))
print("Corrector step size multiplier: {}".format(args.corrector_step_size))

# Override default configs
eval_cfg.sampler.name = 'ConditionalPCTauLeapingAbsorbingInformed'
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
data = dataset.data
test_dataset = np.load(eval_cfg.sampler.test_dataset)
condition_dim = eval_cfg.sampler.condition_dim
descramble_key = np.loadtxt(eval_cfg.pianoroll_dataset_path + '/descramble_key.txt')
# The mask stays the same
descramble_key = np.concatenate([descramble_key, np.array([descramble_key.shape[0]])], axis=0)

def descramble(samples):
    return descramble_key[samples.flatten()].reshape(*samples.shape)

descrambled_test_dataset = descramble(test_dataset)

def get_dist(seq):
    S = 129
    L = seq.shape[0]
    one_hot = np.zeros((L, S))
    seq = np.array(seq, dtype=int)
    one_hot[np.arange(L), seq] = 1
    return np.sum(one_hot, axis=0) / L

def get_mask(seq):
    S = 129
    L = seq.shape[0]
    one_hot = np.zeros((L, S))
    seq = np.array(seq, dtype=int)
    one_hot[np.arange(L), seq] = 1
    return 1 - np.prod(1 - one_hot, axis=0)

def hellinger(seq1, seq2):
    d1, d2 = get_dist(seq1), get_dist(seq2)
    return np.sqrt(.5 * np.sum((d1 ** .5 - d2 ** .5) ** 2))

def outliers(ref, sample):
    ref_mask = get_mask(ref)
    sample_dist = get_dist(sample)
    return np.sum((1 - ref_mask) * sample_dist)

batch_size = 100
num_repeats = 5

sampler = sampling_utils.get_sampler(eval_cfg)
test_size = test_dataset.shape[0]


all_h_dists = []
all_outlier_proportions = []

for _ in range(num_repeats):
    h_dists = []
    outlier_proportions = []
    for start in range(0, test_size, batch_size):
        print(start)
        end = min(start + batch_size, test_size)
        size = end - start
        
        conditioner = torch.from_numpy(test_dataset[start:end, :condition_dim]).to(device)
        samples, _ = sampler.sample(model, size, 1, conditioner)
        samples = descramble(samples)
        for i in range(size):
            h = hellinger(descrambled_test_dataset[start+i, :], samples[i, :])
            r = outliers(descrambled_test_dataset[start+i, :], samples[i, :])
            h_dists.append(h)
            outlier_proportions.append(r)
            
    print("Hellinger distance", np.mean(h_dists))
    print("Proportion of outliers", np.mean(outlier_proportions))
    all_h_dists.append(np.mean(h_dists))
    all_outlier_proportions.append(np.mean(outlier_proportions))

print("------------------------------")
print("Result summary over {} runs:".format(num_repeats))
print("Hellinger distance: {}pm{}".format(np.mean(all_h_dists), np.std(all_h_dists)))
print("Proportion of outliers: {}pm{}".format(np.mean(all_outlier_proportions), 
                                              np.std(all_outlier_proportions)))