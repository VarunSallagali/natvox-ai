import torch
import torch.nn as nn

class NATVOXAdapter(nn.Module):
    def __init__(self):
        super(NATVOXAdapter, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        return x + self.net(x)  # Residual connection