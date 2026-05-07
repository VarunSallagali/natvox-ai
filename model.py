import torch
import torch.nn as nn

class NATVOXAdapter(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=512, dropout=0.1):
        super(NATVOXAdapter, self).__init__()

        self.embedding_dim = embedding_dim

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

        # Learnable residual weighting, initialized to 0.5
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # Ensure input is normalized
        x_norm = nn.functional.normalize(x, p=2, dim=-1)
        
        # Layer 1
        h1 = torch.relu(self.fc1(x_norm))
        h1 = self.dropout(h1)
        h1 = self.ln1(h1)

        # Layer 2
        h2 = torch.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        h2 = self.ln2(h2)

        # Layer 3
        h3 = self.fc3(h2)

        # Residual connection with learnable alpha (scaled by small factor to prevent regression)
        # Alpha should be small to ensure we're transforming, not just returning input
        residual_weight = torch.sigmoid(self.alpha) * 0.2  # Constrain to [0, 0.2]
        out = h3 + residual_weight * x_norm

        # L2 normalization
        out = nn.functional.normalize(out, p=2, dim=-1)

        return out

    def extra_repr(self):
        return f"embedding_dim={self.embedding_dim}, alpha={self.alpha.item():.4f}"