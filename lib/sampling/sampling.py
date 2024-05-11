import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.integrate
import math
from tqdm import tqdm

import lib.sampling.sampling_utils as sampling_utils

def get_initial_samples(N, D, device, S, initial_dist, initial_dist_std=None):
    if initial_dist == 'uniform':
        x = torch.randint(low=0, high=S, size=(N, D), device=device)
    elif initial_dist == 'gaussian':
        target = np.exp(
            - ((np.arange(1, S+1) - S//2)**2) / (2 * initial_dist_std**2)
        )
        target = target / np.sum(target)

        cat = torch.distributions.categorical.Categorical(
            torch.from_numpy(target)
        )
        x = cat.sample((N*D,)).view(N,D)
        x = x.to(device)
    elif initial_dist == 'absorbing':
        x = torch.ones((N, D)) * (S-1)
        x = x.to(device)
    else:
        raise NotImplementedError('Unrecognized initial dist ' + initial_dist)
    return x

def compute_backward(qt0, rate, p0t, in_x, 
                     denom_x=None, eps=1e-9):
    S = rate.shape[-1]
    N, D = in_x.shape
    device = in_x.device
    x_0max = torch.max(p0t, dim=2)[1]

    if denom_x is None:
        # When using the hollow transformer architecture,
        # the model p0t assumes the current location is mask
        # regardless of the true token
        denom_x = in_x

    qt0_denom = qt0[
        torch.arange(N, device=device).repeat_interleave(D*S),
        torch.arange(S, device=device).repeat(N*D),
        denom_x.long().flatten().repeat_interleave(S)
    ].view(N,D,S) + eps   
    # First S is x0 second S is x tilde 
    qt0_numer = qt0 # (N, S, S) 
    forward_rates = rate[
        torch.arange(N, device=device).repeat_interleave(D*S),
        torch.arange(S, device=device).repeat(N*D),
        in_x.long().flatten().repeat_interleave(S)
    ].view(N, D, S) 
    scores = (p0t / qt0_denom) @ qt0_numer
    reverse_rates = forward_rates * scores # (N, D, S)  
    transpose_forward_rates = rate[
        torch.arange(N, device=device).repeat_interleave(D*S),
        in_x.long().flatten().repeat_interleave(S),
        torch.arange(S, device=device).repeat(N*D)
    ].view(N, D, S) 

    return forward_rates, transpose_forward_rates, reverse_rates, x_0max, scores

@sampling_utils.register_sampler
class TauLeaping():
    def __init__(self, cfg):
        self.cfg =cfg

    def sample(self, model, N, num_intermediates):
        t = 1.0
        # C,H,W = self.cfg.data.shape
        # D = C*H*W
        D = np.prod(self.cfg.data.shape)
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(N, D, device, S, initial_dist,
                initial_dist_std)


            ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            counter = 0
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx+1]

                qt0 = model.transition(t * torch.ones((N,), device=device)) # (N, S, S)
                rate = model.rate(t * torch.ones((N,), device=device)) # (N, S, S)

                p0t = F.softmax(model(x, t * torch.ones((N,), device=device)), dim=2) # (N, D, S)


                x_0max = torch.max(p0t, dim=2)[1]
                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())



                qt0_denom = qt0[
                    torch.arange(N, device=device).repeat_interleave(D*S),
                    torch.arange(S, device=device).repeat(N*D),
                    x.long().flatten().repeat_interleave(S)
                ].view(N,D,S) + eps_ratio

                # First S is x0 second S is x tilde

                qt0_numer = qt0 # (N, S, S)

                forward_rates = rate[
                    torch.arange(N, device=device).repeat_interleave(D*S),
                    torch.arange(S, device=device).repeat(N*D),
                    x.long().flatten().repeat_interleave(S)
                ].view(N, D, S)

                inner_sum = (p0t / qt0_denom) @ qt0_numer # (N, D, S)

                reverse_rates = forward_rates * inner_sum # (N, D, S)

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(D),
                    torch.arange(D, device=device).repeat(N),
                    x.long().flatten()
                ] = 0.0

                diffs = torch.arange(S, device=device).view(1,1,S) - x.view(N,D,1)
                poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
                jump_nums = poisson_dist.sample()
                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump
                x_new = torch.clamp(xp, min=0, max=S-1)

                x = x_new

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            p_0gt = F.softmax(model(x, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]
            return x_0max.detach().cpu().numpy().astype(int), x_hist, x0_hist

@sampling_utils.register_sampler
class PCTauLeapingBirthDeath():
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates):
        t = 1.0

        # C,H,W = self.cfg.data.shape
        # D = C*H*W
        D = np.prod(self.cfg.data.shape)
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time
        device = model.device

        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std = model.Q_sigma
        else:
            initial_dist_std = None

        with torch.no_grad():
            x = get_initial_samples(N, D, device, S, initial_dist,
                initial_dist_std)

            h = 1.0 / num_steps # approximately 
            ts = np.linspace(1.0, min_t+h, num_steps)
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            for idx, t in tqdm(enumerate(ts[0:-1])):

                h = ts[idx] - ts[idx+1]

                def get_rates(in_x, in_t):
                    qt0 = model.transition(in_t * torch.ones((N,), device=device)) # (N, S, S)
                    rate = model.rate(in_t * torch.ones((N,), device=device)) # (N, S, S)

                    p0t = F.softmax(model(in_x, in_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)


                    x_0max = torch.max(p0t, dim=2)[1]

                    qt0_denom = qt0[
                        torch.arange(N, device=device).repeat_interleave(D*S),
                        torch.arange(S, device=device).repeat(N*D),
                        in_x.long().flatten().repeat_interleave(S)
                    ].view(N,D,S) + eps_ratio

                    # First S is x0 second S is x tilde

                    qt0_numer = qt0 # (N, S, S)

                    forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(D*S),
                        torch.arange(S, device=device).repeat(N*D),
                        in_x.long().flatten().repeat_interleave(S)
                    ].view(N, D, S)

                    reverse_rates = forward_rates * ((p0t / qt0_denom) @ qt0_numer) # (N, D, S)

                    reverse_rates[
                        torch.arange(N, device=device).repeat_interleave(D),
                        torch.arange(D, device=device).repeat(N),
                        in_x.long().flatten()
                    ] = 0.0

                    transpose_forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(D*S),
                        in_x.long().flatten().repeat_interleave(S),
                        torch.arange(S, device=device).repeat(N*D)
                    ].view(N, D, S)

                    return transpose_forward_rates, reverse_rates, x_0max

                def take_poisson_step(in_x, in_reverse_rates, in_h):
                    diffs = torch.arange(S, device=device).view(1,1,S) - in_x.view(N,D,1)
                    poisson_dist = torch.distributions.poisson.Poisson(in_reverse_rates * in_h)
                    jump_nums = poisson_dist.sample()
                    adj_diffs = jump_nums * diffs
                    overall_jump = torch.sum(adj_diffs, dim=2)
                    unclip_x_new = in_x + overall_jump
                    x_new = torch.clamp(unclip_x_new, min=0, max=S-1)

                    return x_new

                transpose_forward_rates, reverse_rates, x_0max = get_rates(x, t)

                if t in save_ts:
                    x_hist.append(x.detach().cpu().numpy())
                    x0_hist.append(x_0max.detach().cpu().numpy())

                x = take_poisson_step(x, reverse_rates, h)

                if t <= corrector_entry_time:
                    for _ in range(num_corrector_steps):
                        transpose_forward_rates, reverse_rates, _ = get_rates(x, t-h)
                        corrector_rate = transpose_forward_rates + reverse_rates
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(D),
                            torch.arange(D, device=device).repeat(N),
                            x.long().flatten()
                        ] = 0.0
                        x = take_poisson_step(x, corrector_rate, 
                            corrector_step_size_multiplier * h)

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            p_0gt = F.softmax(model(x, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]
            return x_0max.detach().cpu().numpy().astype(int), x_hist, x0_hist
        
@sampling_utils.register_sampler
class PCTauLeapingBarker():
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates):
        t = 1.0

        # C,H,W = self.cfg.data.shape
        # D = C*H*W
        D = np.prod(self.cfg.data.shape)
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time
        device = model.device

        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std = model.Q_sigma
        else:
            initial_dist_std = None

        with torch.no_grad():
            x = get_initial_samples(N, D, device, S, initial_dist,
                initial_dist_std)

            h = 1.0 / num_steps # approximately 
            ts = np.linspace(1.0, min_t+h, num_steps)
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            for idx, t in tqdm(enumerate(ts[0:-1])):

                h = ts[idx] - ts[idx+1]

                def get_rates(in_x, in_t):
                    qt0 = model.transition(in_t * torch.ones((N,), device=device)) # (N, S, S)
                    rate = model.rate(in_t * torch.ones((N,), device=device)) # (N, S, S)

                    p0t = F.softmax(model(in_x, in_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)


                    x_0max = torch.max(p0t, dim=2)[1]

                    qt0_denom = qt0[
                        torch.arange(N, device=device).repeat_interleave(D*S),
                        torch.arange(S, device=device).repeat(N*D),
                        in_x.long().flatten().repeat_interleave(S)
                    ].view(N,D,S) + eps_ratio

                    # First S is x0 second S is x tilde

                    qt0_numer = qt0 # (N, S, S)

                    forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(D*S),
                        torch.arange(S, device=device).repeat(N*D),
                        in_x.long().flatten().repeat_interleave(S)
                    ].view(N, D, S)

                    scores = (p0t / qt0_denom) @ qt0_numer
                    scores[
                        torch.arange(N, device=device).repeat_interleave(D),
                        torch.arange(D, device=device).repeat(N),
                        in_x.long().flatten()
                    ] = 0.0

                    reverse_rates = forward_rates * scores # (N, D, S)

                    transpose_forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(D*S),
                        in_x.long().flatten().repeat_interleave(S),
                        torch.arange(S, device=device).repeat(N*D)
                    ].view(N, D, S)

                    return forward_rates, transpose_forward_rates, reverse_rates, x_0max, scores

                def take_poisson_step(in_x, in_reverse_rates, in_h):
                    diffs = torch.arange(S, device=device).view(1,1,S) - in_x.view(N,D,1)
                    poisson_dist = torch.distributions.poisson.Poisson(in_reverse_rates * in_h)
                    jump_nums = poisson_dist.sample()
                    adj_diffs = jump_nums * diffs
                    overall_jump = torch.sum(adj_diffs, dim=2)
                    unclip_x_new = in_x + overall_jump
                    x_new = torch.clamp(unclip_x_new, min=0, max=S-1)

                    return x_new

                _, _, reverse_rates, x_0max, _ = get_rates(x, t)

                if t in save_ts:
                    x_hist.append(x.detach().cpu().numpy())
                    x0_hist.append(x_0max.detach().cpu().numpy())

                x = take_poisson_step(x, reverse_rates, h)

                if t <= corrector_entry_time:
                    for _ in range(num_corrector_steps):
                        forward_rates, transpose_forward_rates, reverse_rates, _, scores = get_rates(x, t-h)
                        corrector_rate = (transpose_forward_rates + forward_rates) / 2 * scores / (1 + scores)
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(D),
                            torch.arange(D, device=device).repeat(N),
                            x.long().flatten()
                        ] = 0.0
                        x = take_poisson_step(x, corrector_rate, 
                            corrector_step_size_multiplier * h)

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            p_0gt = F.softmax(model(x, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]
            return x_0max.detach().cpu().numpy().astype(int), x_hist, x0_hist
        
@sampling_utils.register_sampler
class PCTauLeapingMPF():
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates):
        t = 1.0

        # C,H,W = self.cfg.data.shape
        # D = C*H*W
        D = np.prod(self.cfg.data.shape)
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time
        device = model.device

        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std = model.Q_sigma
        else:
            initial_dist_std = None

        with torch.no_grad():
            x = get_initial_samples(N, D, device, S, initial_dist,
                initial_dist_std)

            h = 1.0 / num_steps # approximately 
            ts = np.linspace(1.0, min_t+h, num_steps)
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            for idx, t in tqdm(enumerate(ts[0:-1])):

                h = ts[idx] - ts[idx+1]

                def get_rates(in_x, in_t):
                    qt0 = model.transition(in_t * torch.ones((N,), device=device)) # (N, S, S)
                    rate = model.rate(in_t * torch.ones((N,), device=device)) # (N, S, S)

                    p0t = F.softmax(model(in_x, in_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)


                    x_0max = torch.max(p0t, dim=2)[1]

                    qt0_denom = qt0[
                        torch.arange(N, device=device).repeat_interleave(D*S),
                        torch.arange(S, device=device).repeat(N*D),
                        in_x.long().flatten().repeat_interleave(S)
                    ].view(N,D,S) + eps_ratio

                    # First S is x0 second S is x tilde

                    qt0_numer = qt0 # (N, S, S)

                    forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(D*S),
                        torch.arange(S, device=device).repeat(N*D),
                        in_x.long().flatten().repeat_interleave(S)
                    ].view(N, D, S)

                    scores = (p0t / qt0_denom) @ qt0_numer
                    scores[
                        torch.arange(N, device=device).repeat_interleave(D),
                        torch.arange(D, device=device).repeat(N),
                        in_x.long().flatten()
                    ] = 0.0

                    reverse_rates = forward_rates * scores # (N, D, S)

                    transpose_forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(D*S),
                        in_x.long().flatten().repeat_interleave(S),
                        torch.arange(S, device=device).repeat(N*D)
                    ].view(N, D, S)

                    return forward_rates, transpose_forward_rates, reverse_rates, x_0max, scores

                def take_poisson_step(in_x, in_reverse_rates, in_h):
                    diffs = torch.arange(S, device=device).view(1,1,S) - in_x.view(N,D,1)
                    poisson_dist = torch.distributions.poisson.Poisson(in_reverse_rates * in_h)
                    jump_nums = poisson_dist.sample()
                    adj_diffs = jump_nums * diffs
                    overall_jump = torch.sum(adj_diffs, dim=2)
                    unclip_x_new = in_x + overall_jump
                    x_new = torch.clamp(unclip_x_new, min=0, max=S-1)

                    return x_new

                _, _, reverse_rates, x_0max, _ = get_rates(x, t)

                if t in save_ts:
                    x_hist.append(x.detach().cpu().numpy())
                    x0_hist.append(x_0max.detach().cpu().numpy())

                x = take_poisson_step(x, reverse_rates, h)

                if t <= corrector_entry_time:
                    for _ in range(num_corrector_steps):
                        forward_rates, transpose_forward_rates, reverse_rates, _, scores = get_rates(x, t-h)
                        corrector_rate = (transpose_forward_rates + forward_rates) / 2 * torch.sqrt(scores)
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(D),
                            torch.arange(D, device=device).repeat(N),
                            x.long().flatten()
                        ] = 0.0
                        x = take_poisson_step(x, corrector_rate, 
                            corrector_step_size_multiplier * h)

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            p_0gt = F.softmax(model(x, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]
            return x_0max.detach().cpu().numpy().astype(int), x_hist, x0_hist

class PCTauLeapingAbsorbingInformed():
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates):
        t = 1.0

        D = np.prod(self.cfg.data.shape)
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time
        
        if scfg.balancing_function == "barker":
            balancing_function = lambda score: score / (1 + score) 
        elif scfg.balancing_function == "mpf":
            balancing_function = lambda score: torch.sqrt(score)
        elif scfg.balancing_function == "birthdeath":
            balancing_function = None
        else:
            print("Balancing function not found: " + scfg.balancing_function)
            return
        
        device = model.device

        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std = model.Q_sigma
        else:
            initial_dist_std = None

        with torch.no_grad():
            x = get_initial_samples(N, D, device, S, initial_dist,
                initial_dist_std)

            h = 1.0 / num_steps # approximately 
            ts = np.linspace(1.0, min_t+h, num_steps)
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []
            c_rate_hist = []

            for idx, t in tqdm(enumerate(ts[0:-1])):

                h = ts[idx] - ts[idx+1]

                def get_rates(in_x, in_t):
                    qt0 = model.transition(in_t * torch.ones((N,), device=device)) # (N, S, S)
                    rate = model.rate(in_t * torch.ones((N,), device=device)) # (N, S, S)

                    p0t = F.softmax(model(in_x, in_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)

                    denom_x = torch.ones_like(in_x) * (S-1)

                    forward_rates, transpose_forward_rates, reverse_rates, x_0max, scores = compute_backward(qt0, rate, p0t, in_x, denom_x=denom_x, eps=eps_ratio)
                    
                    mask_positions = in_x == (S-1)
                    nonmask_positions = ~mask_positions

                    backward_score_to_curr = scores[
                        torch.arange(N, device=device).repeat_interleave(D),
                        torch.arange(D, device=device).repeat(N),
                        in_x.long().flatten()
                    ].view(N,D)
                    forward_score_from_curr = 1 / (backward_score_to_curr * nonmask_positions + mask_positions)
                    forward_score_from_curr *= nonmask_positions

                    scores = scores * mask_positions.unsqueeze(2)
                    scores[:,:,S-1] = forward_score_from_curr
                    
                    forward_rates[
                        torch.arange(N, device=device).repeat_interleave(D),
                        torch.arange(D, device=device).repeat(N),
                        in_x.long().flatten()
                    ] = 0.0 
                    reverse_rates[
                        torch.arange(N, device=device).repeat_interleave(D),
                        torch.arange(D, device=device).repeat(N),
                        in_x.long().flatten()
                    ] = 0.0 
                    
                    return forward_rates, transpose_forward_rates, reverse_rates, x_0max, scores
                    
                def take_poisson_step(in_x, in_reverse_rates, in_h):
                    diffs = torch.arange(S, device=device).view(1,1,S) - in_x.view(N,D,1)
                    poisson_dist = torch.distributions.poisson.Poisson(in_reverse_rates * in_h)
                    jump_nums = poisson_dist.sample()
                    adj_diffs = jump_nums * diffs
                    overall_jump = torch.sum(adj_diffs, dim=2)
                    unclip_x_new = in_x + overall_jump
                    x_new = torch.clamp(unclip_x_new, min=0, max=S-1)

                    return x_new

                _, _, reverse_rates, x_0max, _ = get_rates(x, t)

                if t in save_ts:
                    x_hist.append(x.detach().cpu().numpy())
                    x0_hist.append(x_0max.detach().cpu().numpy())

                x = take_poisson_step(x, reverse_rates, h)

                if t <= corrector_entry_time:
                    for cstep in range(num_corrector_steps):
                        forward_rates, transpose_forward_rates, reverse_rates, _, scores = get_rates(x, t-h)
                        if balancing_function is None:
                            # We're using the default corrector
                            # which corresponds to birth-death Stein operator
                            corrector_rate = transpose_forward_rates + reverse_rates
                        else:
                            # We removed the one half here because it makes more sense for the absorbing
                            corrector_rate = (transpose_forward_rates + forward_rates) * balancing_function(scores)
                            
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(D),
                            torch.arange(D, device=device).repeat(N),
                            x.long().flatten()
                        ] = 0.0

                        if cstep == 0 and t in save_ts:
                            c_rate_hist.append(corrector_rate.detach().cpu().numpy())

                        x = take_poisson_step(x, corrector_rate, 
                            corrector_step_size_multiplier * h)
                elif t in save_ts:
                    c_rate_hist.append(np.zeros((N, D, S)))

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)
            c_rate_hist = np.array(c_rate_hist)

            p_0gt = F.softmax(model(x, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]

            mask_positions = x == (S-1)
            nonmask_positions = ~mask_positions
            samples = nonmask_positions * x + mask_positions * x_0max
            
            hist = {
                "x": x_hist,
                "x0": x0_hist,
                "rc": c_rate_hist
            }
            
            return samples.detach().cpu().numpy().astype(int), hist

@sampling_utils.register_sampler
class ConditionalTauLeaping():
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates, conditioner):
        assert conditioner.shape[0] == N

        t = 1.0
        condition_dim = self.cfg.sampler.condition_dim
        total_D = self.cfg.data.shape[0]
        sample_D = total_D - condition_dim
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        reject_multiple_jumps = scfg.reject_multiple_jumps
        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(N, sample_D, device, S, initial_dist,
                initial_dist_std)


            ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            counter = 0
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx+1]

                qt0 = model.transition(t * torch.ones((N,), device=device)) # (N, S, S)
                rate = model.rate(t * torch.ones((N,), device=device)) # (N, S, S)

                model_input = torch.concat((conditioner, x), dim=1)
                p0t = F.softmax(model(model_input, t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
                p0t = p0t[:, condition_dim:, :]


                x_0max = torch.max(p0t, dim=2)[1]
                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())



                qt0_denom = qt0[
                    torch.arange(N, device=device).repeat_interleave(sample_D*S),
                    torch.arange(S, device=device).repeat(N*sample_D),
                    x.long().flatten().repeat_interleave(S)
                ].view(N,sample_D,S) + eps_ratio

                # First S is x0 second S is x tilde

                qt0_numer = qt0 # (N, S, S)

                forward_rates = rate[
                    torch.arange(N, device=device).repeat_interleave(sample_D*S),
                    torch.arange(S, device=device).repeat(N*sample_D),
                    x.long().flatten().repeat_interleave(S)
                ].view(N, sample_D, S)

                inner_sum = (p0t / qt0_denom) @ qt0_numer # (N, D, S)

                reverse_rates = forward_rates * inner_sum # (N, D, S)

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(sample_D),
                    torch.arange(sample_D, device=device).repeat(N),
                    x.long().flatten()
                ] = 0.0

                diffs = torch.arange(S, device=device).view(1,1,S) - x.view(N,sample_D,1)
                poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
                jump_nums = poisson_dist.sample()

                if reject_multiple_jumps:
                    jump_num_sum = torch.sum(jump_nums, dim=2)
                    jump_num_sum_mask = jump_num_sum <= 1
                    masked_jump_nums = jump_nums * jump_num_sum_mask.view(N, sample_D, 1)
                    adj_diffs = masked_jump_nums * diffs
                else:
                    adj_diffs = jump_nums * diffs


                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump
                x_new = torch.clamp(xp, min=0, max=S-1)

                x = x_new

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(model(model_input, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int), x_hist, x0_hist


@sampling_utils.register_sampler
class ConditionalPCTauLeapingBirthDeath():
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates, conditioner):
        assert conditioner.shape[0] == N

        t = 1.0
        condition_dim = self.cfg.sampler.condition_dim
        total_D = self.cfg.data.shape[0]
        sample_D = total_D - condition_dim
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        reject_multiple_jumps = scfg.reject_multiple_jumps
        eps_ratio = scfg.eps_ratio

        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time

        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(N, sample_D, device, S, initial_dist,
                initial_dist_std)


            h = 1.0 / num_steps # approximately 
            ts = np.linspace(1.0, min_t+h, num_steps)
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx+1]

                def get_rates(in_x, in_t):
                    qt0 = model.transition(in_t * torch.ones((N,), device=device)) # (N, S, S)
                    rate = model.rate(in_t * torch.ones((N,), device=device)) # (N, S, S)

                    model_input = torch.concat((conditioner, in_x), dim=1)
                    p0t = F.softmax(model(model_input, in_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
                    p0t = p0t[:, condition_dim:, :]


                    x_0max = torch.max(p0t, dim=2)[1]


                    qt0_denom = qt0[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        torch.arange(S, device=device).repeat(N*sample_D),
                        x.long().flatten().repeat_interleave(S)
                    ].view(N,sample_D,S) + eps_ratio

                    # First S is x0 second S is x tilde

                    qt0_numer = qt0 # (N, S, S)

                    forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        torch.arange(S, device=device).repeat(N*sample_D),
                        in_x.long().flatten().repeat_interleave(S)
                    ].view(N, sample_D, S)

                    reverse_rates = forward_rates * ((p0t/qt0_denom) @ qt0_numer) # (N, D, S)

                    reverse_rates[
                        torch.arange(N, device=device).repeat_interleave(sample_D),
                        torch.arange(sample_D, device=device).repeat(N),
                        in_x.long().flatten()
                    ] = 0.0

                    transpose_forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        in_x.long().flatten().repeat_interleave(S),
                        torch.arange(S, device=device).repeat(N*sample_D)
                    ].view(N, sample_D, S)

                    return transpose_forward_rates, reverse_rates, x_0max

                def take_poisson_step(in_x, in_reverse_rates, in_h):
                    diffs = torch.arange(S, device=device).view(1,1,S) - in_x.view(N,sample_D,1)
                    poisson_dist = torch.distributions.poisson.Poisson(in_reverse_rates * in_h)
                    jump_nums = poisson_dist.sample()

                    if reject_multiple_jumps:
                        jump_num_sum = torch.sum(jump_nums, dim=2)
                        jump_num_sum_mask = jump_num_sum <= 1
                        masked_jump_nums = jump_nums * jump_num_sum_mask.view(N, sample_D, 1)
                        adj_diffs = masked_jump_nums * diffs
                    else:
                        adj_diffs = jump_nums * diffs

                    overall_jump = torch.sum(adj_diffs, dim=2)
                    xp = in_x + overall_jump
                    x_new = torch.clamp(xp, min=0, max=S-1)
                    return x_new

                transpose_forward_rates, reverse_rates, x_0max = get_rates(x, t)

                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())

                x = take_poisson_step(x, reverse_rates, h)
                if t <= corrector_entry_time:
                    for _ in range(num_corrector_steps):
                        transpose_forward_rates, reverse_rates, _ = get_rates(x, t-h)
                        corrector_rate = transpose_forward_rates + reverse_rates
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(sample_D),
                            torch.arange(sample_D, device=device).repeat(N),
                            x.long().flatten()
                        ] = 0.0
                        x = take_poisson_step(x, corrector_rate,
                            corrector_step_size_multiplier * h)



            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(model(model_input, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int), x_hist, x0_hist

@sampling_utils.register_sampler
class ConditionalPCTauLeapingBarker():
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates, conditioner):
        assert conditioner.shape[0] == N

        t = 1.0
        condition_dim = self.cfg.sampler.condition_dim
        total_D = self.cfg.data.shape[0]
        sample_D = total_D - condition_dim
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        reject_multiple_jumps = scfg.reject_multiple_jumps
        eps_ratio = scfg.eps_ratio

        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time

        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(N, sample_D, device, S, initial_dist,
                initial_dist_std)


            h = 1.0 / num_steps # approximately 
            ts = np.linspace(1.0, min_t+h, num_steps)
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx+1]

                def get_rates(in_x, in_t):
                    qt0 = model.transition(in_t * torch.ones((N,), device=device)) # (N, S, S)
                    rate = model.rate(in_t * torch.ones((N,), device=device)) # (N, S, S)

                    model_input = torch.concat((conditioner, in_x), dim=1)
                    p0t = F.softmax(model(model_input, in_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
                    p0t = p0t[:, condition_dim:, :]


                    x_0max = torch.max(p0t, dim=2)[1]


                    qt0_denom = qt0[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        torch.arange(S, device=device).repeat(N*sample_D),
                        x.long().flatten().repeat_interleave(S)
                    ].view(N,sample_D,S) + eps_ratio

                    # First S is x0 second S is x tilde

                    qt0_numer = qt0 # (N, S, S)

                    forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        torch.arange(S, device=device).repeat(N*sample_D),
                        in_x.long().flatten().repeat_interleave(S)
                    ].view(N, sample_D, S)

                    scores = (p0t / qt0_denom) @ qt0_numer
                    scores[
                        torch.arange(N, device=device).repeat_interleave(sample_D),
                        torch.arange(sample_D, device=device).repeat(N),
                        in_x.long().flatten()
                    ] = 0.0

                    reverse_rates = forward_rates * scores # (N, D, S)

                    transpose_forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        in_x.long().flatten().repeat_interleave(S),
                        torch.arange(S, device=device).repeat(N*sample_D)
                    ].view(N, sample_D, S)

                    return transpose_forward_rates, reverse_rates, x_0max, scores

                def take_poisson_step(in_x, in_reverse_rates, in_h):
                    diffs = torch.arange(S, device=device).view(1,1,S) - in_x.view(N,sample_D,1)
                    poisson_dist = torch.distributions.poisson.Poisson(in_reverse_rates * in_h)
                    jump_nums = poisson_dist.sample()

                    if reject_multiple_jumps:
                        jump_num_sum = torch.sum(jump_nums, dim=2)
                        jump_num_sum_mask = jump_num_sum <= 1
                        masked_jump_nums = jump_nums * jump_num_sum_mask.view(N, sample_D, 1)
                        adj_diffs = masked_jump_nums * diffs
                    else:
                        adj_diffs = jump_nums * diffs

                    overall_jump = torch.sum(adj_diffs, dim=2)
                    xp = in_x + overall_jump
                    x_new = torch.clamp(xp, min=0, max=S-1)
                    return x_new

                transpose_forward_rates, reverse_rates, x_0max, _ = get_rates(x, t)

                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())

                x = take_poisson_step(x, reverse_rates, h)
                if t <= corrector_entry_time:
                    for _ in range(num_corrector_steps):
                        transpose_forward_rates, reverse_rates, _, scores = get_rates(x, t-h)
                        corrector_rate = transpose_forward_rates * scores / (1 + scores)
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(sample_D),
                            torch.arange(sample_D, device=device).repeat(N),
                            x.long().flatten()
                        ] = 0.0
                        x = take_poisson_step(x, corrector_rate,
                            corrector_step_size_multiplier * h)



            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(model(model_input, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int), x_hist, x0_hist
        
@sampling_utils.register_sampler
class ConditionalPCTauLeapingMPF():
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates, conditioner):
        assert conditioner.shape[0] == N

        t = 1.0
        condition_dim = self.cfg.sampler.condition_dim
        total_D = self.cfg.data.shape[0]
        sample_D = total_D - condition_dim
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        reject_multiple_jumps = scfg.reject_multiple_jumps
        eps_ratio = scfg.eps_ratio

        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time

        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(N, sample_D, device, S, initial_dist,
                initial_dist_std)


            h = 1.0 / num_steps # approximately 
            ts = np.linspace(1.0, min_t+h, num_steps)
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx+1]

                def get_rates(in_x, in_t):
                    qt0 = model.transition(in_t * torch.ones((N,), device=device)) # (N, S, S)
                    rate = model.rate(in_t * torch.ones((N,), device=device)) # (N, S, S)

                    model_input = torch.concat((conditioner, in_x), dim=1)
                    p0t = F.softmax(model(model_input, in_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
                    p0t = p0t[:, condition_dim:, :]


                    x_0max = torch.max(p0t, dim=2)[1]


                    qt0_denom = qt0[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        torch.arange(S, device=device).repeat(N*sample_D),
                        x.long().flatten().repeat_interleave(S)
                    ].view(N,sample_D,S) + eps_ratio

                    # First S is x0 second S is x tilde

                    qt0_numer = qt0 # (N, S, S)

                    forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        torch.arange(S, device=device).repeat(N*sample_D),
                        in_x.long().flatten().repeat_interleave(S)
                    ].view(N, sample_D, S)

                    scores = (p0t / qt0_denom) @ qt0_numer
                    scores[
                        torch.arange(N, device=device).repeat_interleave(sample_D),
                        torch.arange(sample_D, device=device).repeat(N),
                        in_x.long().flatten()
                    ] = 0.0

                    reverse_rates = forward_rates * scores # (N, D, S)

                    transpose_forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        in_x.long().flatten().repeat_interleave(S),
                        torch.arange(S, device=device).repeat(N*sample_D)
                    ].view(N, sample_D, S)

                    return transpose_forward_rates, reverse_rates, x_0max, scores

                def take_poisson_step(in_x, in_reverse_rates, in_h):
                    diffs = torch.arange(S, device=device).view(1,1,S) - in_x.view(N,sample_D,1)
                    poisson_dist = torch.distributions.poisson.Poisson(in_reverse_rates * in_h)
                    jump_nums = poisson_dist.sample()

                    if reject_multiple_jumps:
                        jump_num_sum = torch.sum(jump_nums, dim=2)
                        jump_num_sum_mask = jump_num_sum <= 1
                        masked_jump_nums = jump_nums * jump_num_sum_mask.view(N, sample_D, 1)
                        adj_diffs = masked_jump_nums * diffs
                    else:
                        adj_diffs = jump_nums * diffs

                    overall_jump = torch.sum(adj_diffs, dim=2)
                    xp = in_x + overall_jump
                    x_new = torch.clamp(xp, min=0, max=S-1)
                    return x_new

                transpose_forward_rates, reverse_rates, x_0max, _ = get_rates(x, t)

                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())

                x = take_poisson_step(x, reverse_rates, h)
                if t <= corrector_entry_time:
                    for _ in range(num_corrector_steps):
                        transpose_forward_rates, reverse_rates, _, scores = get_rates(x, t-h)
                        corrector_rate = transpose_forward_rates * torch.sqrt(scores)
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(sample_D),
                            torch.arange(sample_D, device=device).repeat(N),
                            x.long().flatten()
                        ] = 0.0
                        x = take_poisson_step(x, corrector_rate,
                            corrector_step_size_multiplier * h)



            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(model(model_input, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int), x_hist, x0_hist