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
        # Layer 1
        h1 = torch.relu(self.fc1(x))
        h1 = self.dropout(h1)
        h1 = self.ln1(h1)

        # Layer 2
        h2 = torch.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        h2 = self.ln2(h2)

        # Layer 3
        h3 = self.fc3(h2)

        # Residual connection with learnable alpha
        out = h3 + self.alpha * x

        # L2 normalization
        out = nn.functional.normalize(out, p=2, dim=-1)

        return out

    def extra_repr(self):
        return f"embedding_dim={self.embedding_dim}, alpha={self.alpha.item():.4f}"