import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
import sys
import os
sys.path.append(os.getcwd())
from config.eval.cifar10 import get_config as get_eval_config
import lib.utils.bookkeeping as bookkeeping
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import lib.utils.utils as utils
import lib.models.models as models
import lib.models.model_utils as model_utils
import lib.sampling.sampling as sampling
import lib.sampling.sampling_utils as sampling_utils
from PIL import Image

import argparse

# Create the parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--sampler', type=str, default="BirthDeath", help='')
parser.add_argument('--corrector-steps', type=int, default=10, help='')
parser.add_argument('--entry-time', type=float, default=0.1, help='')
parser.add_argument('--corrector-step-size', type=float, default=1.5, help='')

# Parse the arguments
args = parser.parse_args()

save_samples_path = '/scratch/users/yixiuz/model_samples/cifar/' + args.sampler + "-" + str(args.corrector_steps) + "-stepsize-" + str(args.corrector_step_size)

os.makedirs(save_samples_path, exist_ok=True)
print("Sampler: {}".format(args.sampler))
print("Number of corrector steps: {}".format(args.corrector_steps))
print("Corrector entry time: {}".format(args.entry_time))
print("Corrector step size multiplier: {}".format(args.corrector_step_size))

eval_cfg = get_eval_config()
train_cfg = bookkeeping.load_ml_collections(Path(eval_cfg.train_config_path))

for item in eval_cfg.train_config_overrides:
    utils.set_in_nested_dict(train_cfg, item[0], item[1])

# Override default configs
eval_cfg.sampler.name = 'PCTauLeaping' + args.sampler
eval_cfg.sampler.num_corrector_steps = args.corrector_steps
eval_cfg.sampler.corrector_entry_time = args.entry_time
eval_cfg.sampler.corrector_step_size_multiplier = args.corrector_step_size

S = train_cfg.data.S
device = torch.device(eval_cfg.device)

model = model_utils.create_model(train_cfg, device)

loaded_state = torch.load(Path(eval_cfg.checkpoint_path),
    map_location=device)

modified_model_state = utils.remove_module_from_keys(loaded_state['model'])
model.load_state_dict(modified_model_state)

model.eval()

def imgtrans(x):
    x = np.transpose(x, (1,2,0))
    return x

total_samples = 0
batch = 50
sampler = sampling_utils.get_sampler(eval_cfg)

while True:
    print(total_samples)
    samples, _, _ = sampler.sample(model, batch, 1)
    samples = samples.reshape(batch, 3, 32, 32)
    samples_uint8 = samples.astype(np.uint8)
    for i in range(samples.shape[0]):
        path_to_save = save_samples_path + f'/{total_samples + i}.png'
        img = Image.fromarray(imgtrans(samples_uint8[i]))
        img.save(path_to_save)


    total_samples += batch
    if total_samples >= 50000:
        break
