import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import get_data


# -----------------------------
# LOSS FUNCTION AS PER PAPER
# -----------------------------
def adapter_loss(adapted_emb, target_emb, synthetic_emb, model, lambda1=0.1, lambda2=1e-4):
    # Cosine distance
    cosine_dist = 1 - F.cosine_similarity(adapted_emb, target_emb, dim=-1).mean()
    
    # Euclidean distance (L2 norm)
    euclidean_dist = torch.norm(adapted_emb - target_emb, p=2, dim=-1).mean()
    
    # L2 weight regularization (sum of squared weights)
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2)
    reg_loss *= lambda2
    
    total_loss = cosine_dist + lambda1 * euclidean_dist + reg_loss
    
    return total_loss, cosine_dist, euclidean_dist


# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train(model, synthetic=None, natural=None, epochs=50, batch_size=5, lr=1e-3, checkpoint_path="natvox_model.pth"):

    # Load data
    if synthetic is None or natural is None:
        synthetic, natural = get_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    synthetic = synthetic.to(device)
    natural = natural.to(device)

    dataset = torch.utils.data.TensorDataset(synthetic, natural)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    history = {
        'total_loss': [],
        'cosine_loss': [],
        'l2_loss': [],
        'sim': []
    }

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    for epoch in range(1, epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_cos = 0.0
        epoch_l2 = 0.0
        epoch_sim = 0.0

        for synthetic_batch, natural_batch in dataloader:

            synthetic_batch = synthetic_batch.to(device)
            natural_batch = natural_batch.to(device)

            # 🔥 NORMALIZE INPUTS (VERY IMPORTANT)
            synthetic_batch = F.normalize(synthetic_batch, p=2, dim=-1)
            natural_batch = F.normalize(natural_batch, p=2, dim=-1)

            # Forward
            adapted = model(synthetic_batch)

            # Loss
            loss, cosl, l2l = adapter_loss(adapted, natural_batch, synthetic_batch, model)

            optimizer.zero_grad()
            loss.backward()

            # 🔥 GRADIENT CLIPPING
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Stats
            batch_size_actual = synthetic_batch.size(0)

            epoch_loss += loss.item() * batch_size_actual
            epoch_cos += cosl.item() * batch_size_actual
            epoch_l2 += l2l.item() * batch_size_actual

            with torch.no_grad():
                epoch_sim += F.cosine_similarity(
                    adapted, natural_batch, dim=-1
                ).sum().item()

        # Averages
        n = len(dataset)
        avg_loss = epoch_loss / n
        avg_cos = epoch_cos / n
        avg_l2 = epoch_l2 / n
        avg_sim = epoch_sim / n

        history['total_loss'].append(avg_loss)
        history['cosine_loss'].append(avg_cos)
        history['l2_loss'].append(avg_l2)
        history['sim'].append(avg_sim)

        scheduler.step(avg_loss)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            alpha_val = model.alpha.item() if hasattr(model, 'alpha') else 0
            print(
                f"Epoch {epoch}/{epochs} | "
                f"loss={avg_loss:.4f} | "
                f"cos={avg_cos:.4f} | "
                f"l2={avg_l2:.4f} | "
                f"sim={avg_sim:.4f} | "
                f"alpha={alpha_val:.4f}"
            )

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    try:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

    return model, history


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    from model import NATVOXAdapter

    model = NATVOXAdapter()
    train(model)