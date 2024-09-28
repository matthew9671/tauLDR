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

# poisson step with rejection
def take_poisson_step(in_x, rates, h):
    # Generate Poisson distributed jumps for each rate
    poisson_dist = torch.distributions.poisson.Poisson(rates * h)
    jump_nums = poisson_dist.sample()
        
    # Get the index of the maximum jump
    jump_target = torch.argmax(jump_nums, dim=-1)
    # Create the mask to decide whether to jump or not
    out = torch.where(jump_nums.sum(dim=-1) == 1, jump_target, in_x)
    
    return out

def take_euler_step(in_x, rates, h):
    N, D = in_x.shape
    
    rates *= h
    
    # Sum of the rates for each row
    sum_rates = torch.sum(rates, dim=-1)
    
    # Transition logit: log(1 - exp(-rates)) = log(-expm1(-rates))
    transition_logit = torch.log(-torch.expm1(-rates))
    
    # Set the diagonal of transition_logit to -sum_rates
    transition_logit[torch.arange(N, device=device).repeat_interleave(D),
                     torch.arange(D, device=device).repeat(N),
                     in_x.long().flatten()] = -sum_rates.flatten()
    
    # Use categorical sampling to select the next state
    dist = torch.distributions.categorical.Categorical(logits=transition_logit)
    out = dist.sample().long()
    
    return out

@sampling_utils.register_sampler
class PCKGillespies():
    def __init__(self, cfg):
        self.cfg =cfg

    def sample(self, model, N, num_intermediates=0):
        t = 1.0
        D = np.prod(self.cfg.data.shape)
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        initial_dist = scfg.initial_dist
        
        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time
        
        updates_per_eval = scfg.updates_per_eval

        if scfg.balancing_function == "barker":
            balancing_function = lambda score: score / (1 + score) 
        elif scfg.balancing_function == "mpf":
            balancing_function = lambda score: torch.sqrt(score)
        elif scfg.balancing_function == "birthdeath":
            balancing_function = None
        else:
            print("Balancing function not found: " + scfg.balancing_function)
            return
        
        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        # Now the batch is not syncronized anymore
        ts = t * torch.ones((N,), device=device)
        update_mask = ts > min_t
        num_updates = 0
        
        with torch.no_grad():
            x = get_initial_samples(N, D, device, S, initial_dist,
                initial_dist_std)
            
            pbar = tqdm(total=D)
            while num_updates < D:

                # Compute backward transition rate
                qt0 = model.transition(ts) # (N, S, S)
                rate = model.rate(ts) # (N, S, S)

                p0t = F.softmax(model(x, ts), dim=2) # (N, D, S)

                Rf, RfT, Rb, x_0max, scores = compute_backward(qt0, rate, p0t, x)
                # Rb: (N, D, S)
                Rb[torch.arange(N, device=device).repeat_interleave(D),
                   torch.arange(D, device=device).repeat(N),
                   x.long().flatten()] = eps_ratio
            
                # Compute total rate (N, D)
                Rb_sum = torch.sum(Rb, axis=2)
                # Sample a holding time (N, D)
                taus = torch.distributions.Exponential(Rb_sum).sample()
                # Find the position of the shortest holding time for each dimension (N,)
                ids_sorted = torch.argsort(taus, axis=1)
                dts = torch.zeros((N,), device=device)
                
                for update in range(updates_per_eval):
                    # Make one round of updates
                    ids = ids_sorted[:, update]
                    # (N,)
                    dts = taus[torch.arange(N, device=device), ids] - dts

                    # Rates given the dimensions of transition (N, S)
                    rates_single = Rb[torch.arange(N, device=device),ids]
                    # Total rate given the dimensions of transition (N, 1)
                    rates_sum_single = torch.unsqueeze(Rb_sum[torch.arange(N, device=device),ids], 1)
                    # The targets of transition (N,)
                    updates = torch.multinomial(rates_single / rates_sum_single, 1)[:,0]
                    # Update ts
                    update_mask = update_mask & ((ts - dts * update_mask) > min_t)
                    ts -= dts * update_mask
                    # Update x
                    original = x[torch.arange(N, device=device), ids]
                    x[torch.arange(N, device=device), ids] = updates * update_mask + original * (~update_mask)
                    num_updates += 1
                    pbar.update(1)
                    
                def get_rates(in_x, in_t):
                    qt0 = model.transition(in_t) # (N, S, S)
                    rate = model.rate(in_t) # (N, S, S)

                    p0t = F.softmax(model(in_x, in_t), dim=2) # (N, D, S)

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

                if num_updates >= (1-corrector_entry_time) * D:
                    
                    h = 1/D
                    
                    for cstep in range(num_corrector_steps):
                        forward_rates, transpose_forward_rates, reverse_rates, _, scores = get_rates(x, ts) # ts-h?
                        if balancing_function is None:
                            # We're using the default corrector
                            # which corresponds to birth-death Stein operator
                            corrector_rate = transpose_forward_rates + reverse_rates
                        else:
                            # We removed the one half here because it makes more sense for the absorbing
                            corrector_rate = (transpose_forward_rates + forward_rates) * balancing_function(scores)
                        # Only update dimensions with 
                        corrector_rate *= update_mask.unsqueeze(1).unsqueeze(1)
                            
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(D),
                            torch.arange(D, device=device).repeat(N),
                            x.long().flatten()
                        ] = 0.0

                        x = take_euler_step(x, corrector_rate, 
                            corrector_step_size_multiplier * h)

            p_0gt = F.softmax(model(x, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]

            mask_positions = x == (S-1)
            nonmask_positions = ~mask_positions
            samples = nonmask_positions * x + mask_positions * x_0max
            
            out = {
                "ts": ts.cpu().numpy()
            }
            
            return samples.detach().cpu().numpy().astype(int), out
        
@sampling_utils.register_sampler
class ConditionalPCKGillespies():
    def __init__(self, cfg):
        self.cfg =cfg

    def sample(self, model, N, num_intermediates, conditioner):
        assert conditioner.shape[0] == N
        
        condition_dim = self.cfg.sampler.condition_dim
        total_D = np.prod(self.cfg.data.shape)
        D = total_D - condition_dim
        
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        # num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        initial_dist = scfg.initial_dist
        # Specific to Gillespies
        updates_per_eval = scfg.updates_per_eval
        
        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device
        
        # Corrector stuff
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

        # Now the batch is not syncronized anymore
        ts = 1.0 * torch.ones((N,), device=device)
        update_mask = ts > min_t
        num_updates = 0
        
        with torch.no_grad():
            x = get_initial_samples(N, D, device, S, initial_dist,
                initial_dist_std)

            # x_hist = []
            # x0_hist = []

            pbar = tqdm(total=D)
            while num_updates < D:

                # Compute backward transition rate
                qt0 = model.transition(ts) # (N, S, S)
                rate = model.rate(ts) # (N, S, S)

                model_input = torch.concat((conditioner, x), dim=1)
                p0t = F.softmax(model(model_input, ts), dim=2) # (N, D, S)
                p0t = p0t[:, condition_dim:, :]

                Rf, RfT, Rb, x_0max, scores = compute_backward(qt0, rate, p0t, x)
                # Rb: (N, D, S)
                Rb[torch.arange(N, device=device).repeat_interleave(D),
                   torch.arange(D, device=device).repeat(N),
                   x.long().flatten()] = eps_ratio
            
                # Compute total rate (N, D)
                Rb_sum = torch.sum(Rb, axis=2)
                # Sample a holding time (N, D)
                taus = torch.distributions.Exponential(Rb_sum).sample()
                # Find the position of the shortest holding time for each dimension (N,)
                ids_sorted = torch.argsort(taus, axis=1)
                dts = torch.zeros((N,), device=device)
                
                for update in range(updates_per_eval):
                    # Make one round of updates
                    ids = ids_sorted[:, update]
                    # (N,)
                    dts = taus[torch.arange(N, device=device), ids] - dts

                    # Rates given the dimensions of transition (N, S)
                    rates_single = Rb[torch.arange(N, device=device),ids]
                    # Total rate given the dimensions of transition (N, 1)
                    rates_sum_single = torch.unsqueeze(Rb_sum[torch.arange(N, device=device),ids], 1)
                    # The targets of transition (N,)
                    updates = torch.multinomial(rates_single / rates_sum_single, 1)[:,0]
                    # Update ts
                    update_mask = update_mask & ((ts - dts * update_mask) > min_t)
                    ts -= dts * update_mask
                    # Update x
                    original = x[torch.arange(N, device=device), ids]
                    x[torch.arange(N, device=device), ids] = updates * update_mask + original * (~update_mask)
                    num_updates += 1
                    pbar.update(1)
                    
                # Corrector time
                def get_rates(in_x, in_t):
                    qt0 = model.transition(in_t) # (N, S, S)
                    rate = model.rate(in_t) # (N, S, S)

                    p0t = F.softmax(model(in_x, in_t), dim=2) # (N, D, S)

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

                if num_updates >= (1-corrector_entry_time) * D:
                    
                    h = 1/D
                    
                    for cstep in range(num_corrector_steps):
                        forward_rates, transpose_forward_rates, reverse_rates, _, scores = get_rates(x, ts) # ts-h?
                        if balancing_function is None:
                            # We're using the default corrector
                            # which corresponds to birth-death Stein operator
                            corrector_rate = transpose_forward_rates + reverse_rates
                        else:
                            # We removed the one half here because it makes more sense for the absorbing
                            corrector_rate = (transpose_forward_rates + forward_rates) * balancing_function(scores)
                        # Only update dimensions with 
                        corrector_rate *= update_mask.unsqueeze(1).unsqueeze(1)
                            
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(D),
                            torch.arange(D, device=device).repeat(N),
                            x.long().flatten()
                        ] = eps_ratio
                        
                        x = take_euler_step(x, corrector_rate, 
                            corrector_step_size_multiplier * h)
                
                
            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(model(model_input, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            
            x_0max = torch.max(p_0gt, dim=2)[1]

            mask_positions = x == (S-1)
            nonmask_positions = ~mask_positions
            samples = nonmask_positions * x + mask_positions * x_0max
            
            output = torch.concat((conditioner, samples), dim=1)
            
            out = {
                "ts": ts.cpu().numpy()
            }
            
            return output.detach().cpu().numpy().astype(int), out

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
class ConditionalPCTauLeapingAbsorbingInformed():
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates, conditioner):
        assert conditioner.shape[0] == N

        t = 1.0
        condition_dim = self.cfg.sampler.condition_dim
        total_D = np.prod(self.cfg.data.shape)
        sample_D = total_D - condition_dim
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
            x = get_initial_samples(N, sample_D, device, S, initial_dist,
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

                    model_input = torch.concat((conditioner, in_x), dim=1)
                    p0t = F.softmax(model(model_input, in_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
                    p0t = p0t[:, condition_dim:, :]

                    denom_x = torch.ones_like(in_x) * (S-1)

                    forward_rates, transpose_forward_rates, reverse_rates, x_0max, scores = compute_backward(qt0, rate, p0t, in_x, denom_x=denom_x, eps=eps_ratio)
                    
                    mask_positions = in_x == (S-1)
                    nonmask_positions = ~mask_positions

                    backward_score_to_curr = scores[
                        torch.arange(N, device=device).repeat_interleave(sample_D),
                        torch.arange(sample_D, device=device).repeat(N),
                        in_x.long().flatten()
                    ].view(N,sample_D)
                    forward_score_from_curr = 1 / (backward_score_to_curr * nonmask_positions + mask_positions)
                    forward_score_from_curr *= nonmask_positions

                    scores = scores * mask_positions.unsqueeze(2)
                    scores[:,:,S-1] = forward_score_from_curr
                    
                    forward_rates[
                        torch.arange(N, device=device).repeat_interleave(sample_D),
                        torch.arange(sample_D, device=device).repeat(N),
                        in_x.long().flatten()
                    ] = 0.0 
                    reverse_rates[
                        torch.arange(N, device=device).repeat_interleave(sample_D),
                        torch.arange(sample_D, device=device).repeat(N),
                        in_x.long().flatten()
                    ] = 0.0 
                    
                    return forward_rates, transpose_forward_rates, reverse_rates, x_0max, scores

                _, _, reverse_rates, x_0max, _ = get_rates(x, t)

                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())

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
                            torch.arange(N, device=device).repeat_interleave(sample_D),
                            torch.arange(sample_D, device=device).repeat(N),
                            x.long().flatten()
                        ] = 0.0

                        if cstep == 0 and t in save_ts:
                            c_rate_hist.append(corrector_rate.detach().cpu().numpy())

                        x = take_poisson_step(x, corrector_rate, 
                            corrector_step_size_multiplier * h)
                elif t in save_ts:
                    c_rate_hist.append(np.zeros((N, sample_D, S)))

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)
            c_rate_hist = np.array(c_rate_hist)

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(model(model_input, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]

            x_0max = torch.max(p_0gt, dim=2)[1]

            mask_positions = x == (S-1)
            nonmask_positions = ~mask_positions
            samples = nonmask_positions * x + mask_positions * x_0max

            output = torch.concat((conditioner, samples), dim=1)
            
            hist = {
                "x": x_hist,
                "x0": x0_hist,
                "rc": c_rate_hist
            }
            
            return output.detach().cpu().numpy().astype(int), hist
        
@sampling_utils.register_sampler
class PCEulerAbsorbingInformed():
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

                _, _, reverse_rates, x_0max, _ = get_rates(x, t)

                if t in save_ts:
                    x_hist.append(x.detach().cpu().numpy())
                    x0_hist.append(x_0max.detach().cpu().numpy())

                x = take_euler_step(x, reverse_rates, h)

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

                        x = take_euler_step(x, corrector_rate, 
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