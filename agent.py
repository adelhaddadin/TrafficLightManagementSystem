import torch.nn as nn

# Constants
POSSIBLE_DURATIONS = [5, 10, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
NUM_DURATIONS = len(POSSIBLE_DURATIONS)

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
