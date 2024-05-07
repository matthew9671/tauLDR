import torch
import torch.nn as nn
import math
import lib.networks.network_utils as network_utils
from lib.networks.networks import PositionalEncoding, FFResidual
import torch.nn.functional as F
import numpy as np

class DirectionalTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, temb_dim, 
                 kdim=None, vdim=None, direction='forward'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads,
            dropout=dropout, batch_first=True, kdim=kdim, vdim=vdim)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.film_from_temb = nn.Linear(temb_dim, 2*d_model)

        self.direction = direction

    def forward(self,
        query, # ["B", "L", "K"],
        key_value,
        temb, # ["B", "temb_dim"]
    ):
        q, kv = query, key_value
        B, L, K = q.shape
        film_params = self.film_from_temb(temb)

        x = self.norm1(q + self._sa_block(q, kv))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]
        x = self.norm2(x + self._ff_block(x))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]

        return x

    def _sa_block(self, query, key_value):
        B, L, K = query.shape
        device = query.device
        # Construct the mask here
        # Maybe we can pre-compute this mask and store it in the model
        if self.direction == 'forward':
            mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1).to(device)
        elif self.direction == 'backward':
            mask = torch.tril(torch.ones(L, L) * float('-inf'), diagonal=-1).to(device)
        elif self.direction == 'mixed':
            # concatenate both the forward mask and the backward mask
            # this time mask out the diagonals
            forward_mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1).to(device)
            backward_mask = torch.tril(torch.ones(L, L) * float('-inf'), diagonal=-1).to(device)
            mask = torch.cat([forward_mask, backward_mask], dim=1)
        else:
            raise ValueError('Invalid direction')
        
        x = self.self_attn(query, key_value, key_value, attn_mask=mask)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class HollowTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward,
        dropout, num_output_FFresiduals, time_scale_factor, S, max_len,
        temb_dim, use_one_hot_input, num_layers_per_mixed, device):
        super().__init__()

        self.temb_dim = temb_dim
        self.use_one_hot_input = use_one_hot_input
        self.num_layers_per_mixed = num_layers_per_mixed

        self.S = S

        self.pos_embed = PositionalEncoding(device, d_model, dropout, max_len)

        self.forward_encoder_layers = []
        self.backward_encoder_layers = []
        self.mixer_encoder_layers = []
        for i in range(num_layers):
            self.forward_encoder_layers.append(
                DirectionalTransformerEncoderLayer(d_model, num_heads, dim_feedforward,
                    dropout, 4*temb_dim, direction='forward')
            )
            self.backward_encoder_layers.append(
                DirectionalTransformerEncoderLayer(d_model, num_heads, dim_feedforward,
                    dropout, 4*temb_dim, direction='backward')
            )
            if (i + 1) % num_layers_per_mixed == 0:
                self.mixer_encoder_layers.append(
                    DirectionalTransformerEncoderLayer(d_model * 2, num_heads, dim_feedforward,
                        dropout, 4*temb_dim, kdim=d_model, vdim=d_model, direction='mixed')
                )
        self.forward_encoder_layers = nn.ModuleList(self.forward_encoder_layers)
        self.backward_encoder_layers = nn.ModuleList(self.backward_encoder_layers)
        self.mixer_encoder_layers = nn.ModuleList(self.mixer_encoder_layers)

        self.output_resid_layers = []
        for i in range(num_output_FFresiduals):
            self.output_resid_layers.append(
                FFResidual(d_model * 2, dim_feedforward, 4*temb_dim)
            )
        self.output_resid_layers = nn.ModuleList(self.output_resid_layers)

        self.output_linear = nn.Linear(d_model * 2, self.S)
        
        if use_one_hot_input:
            self.input_embedding = nn.Linear(S, d_model)
        else:
            self.input_embedding = nn.Linear(1, d_model)

        self.temb_net = nn.Sequential(
            nn.Linear(temb_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 4*temb_dim)
        )

        self.time_scale_factor = time_scale_factor

    def forward(self, x, # ["B", "L"],
        times #["B"]
    ):
        B, L = x.shape

        temb = self.temb_net(
            network_utils.transformer_timestep_embedding(
                times*self.time_scale_factor, self.temb_dim
            )
        )
        one_hot_x = nn.functional.one_hot(x, num_classes=self.S) # (B, L, S)

        if self.use_one_hot_input:
            x = self.input_embedding(one_hot_x.float()) # (B, L, K)
        else:
            x = self.normalize_input(x)
            x = x.view(B, L, 1)
            x = self.input_embedding(x) # (B, L, K)

        x = self.pos_embed(x)

        zero_pad = torch.zeros(x.size(0), 1, x.size(2)).to(x.device)
        forward_x = torch.cat([zero_pad, x[:,:-1]], dim=1)
        backward_x = torch.cat([x[:,1:], zero_pad], dim=1)
        mixed_x = None

        # for encoder_layer in self.encoder_layers:
        for i in range(len(self.forward_encoder_layers)):
            forward_x = self.forward_encoder_layers[i](forward_x, forward_x, temb)
            backward_x = self.backward_encoder_layers[i](backward_x, backward_x, temb)
            if (i+1) % self.num_layers_per_mixed == 0:
                concat_output = torch.cat([forward_x, backward_x], dim=1)
                if mixed_x is None:
                    mixed_x = torch.cat([forward_x, backward_x], dim=2)
                mixed_x = self.mixer_encoder_layers[i // self.num_layers_per_mixed](mixed_x, concat_output, temb)
        x = mixed_x

        # x (B, L, K)
        for resid_layer in self.output_resid_layers:
            x = resid_layer(x, temb)

        x = self.output_linear(x) # (B, L, S)

        # We can't let information of the current position be leaked
        # x = x + one_hot_x

        return x

    def normalize_input(self, x):
        x = x/self.S # (0, 1)
        x = x*2 - 1 # (-1, 1)
        return x