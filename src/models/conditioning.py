import torch
import torch.nn as nn


class FiLMConditioner(nn.Module):

    def __init__(self, hidden):

        super().__init__()

        self.gamma = nn.Linear(hidden, hidden)
        self.beta = nn.Linear(hidden, hidden)

    def forward(self, h_claim, H_u, sample_map):
    
        sample_map = sample_map.long()

        H_case = H_u[sample_map]

        gamma = self.gamma(H_case)
        beta = self.beta(H_case)

        h_cond = gamma * h_claim + beta

        return h_cond