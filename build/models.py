# model.py

import torch
import torch.nn as nn
import numpy as np
from replay_buffer import PrioritizedReplayBuffer  # Import the class

from rl_helpers import NUM_DURATIONS

class EnhancedDQNAgent(nn.Module):
    def __init__(self, state_size: int, num_tl: int, num_phases: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(128, num_tl * num_phases * NUM_DURATIONS)

    def forward(self, x):
        feat = self.network(x)
        q_vals = self.output_layer(feat)
        return q_vals
