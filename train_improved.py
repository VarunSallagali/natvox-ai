"""
Improved training script with validation and early stopping.
This ensures the model actually learns to IMPROVE similarity, not degrade it.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import get_data
from model import NATVOXAdapter


def adapter_loss(adapted_emb, target_emb, synthetic_emb, model, lambda1=0.1, lambda2=1e-4):
    """Loss function that explicitly maximizes similarity to target."""
    # Cosine similarity (we want this to be HIGH, so minimize negative)
    cosine_sim = F.cosine_similarity(adapted_emb, target_emb, dim=-1).mean()
    cosine_loss = 1.0 - cosine_sim  # Convert similarity to loss
    
    # Euclidean distance (L2 norm)
    euclidean_dist = torch.norm(adapted_emb - target_emb, p=2, dim=-1).mean()
    
    # L2 weight regularization
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2)
    reg_loss *= lambda2
    
    total_loss = cosine_loss + lambda1 * euclidean_dist + reg_loss
    
    return total_loss, cosine_loss, euclidean_dist


def validate(model, synthetic, natural, device):
    """Validation function to check if model is improving."""
    model.eval()
    with torch.no_grad():
        synthetic = synthetic.to(device)
        natural = natural.to(device)
        
        adapted = model(synthetic)
        
        # Calculate improvement
        cos_sim = F.cosine_similarity(synthetic, natural, dim=-1).mean()
        cos_sim_adapted = F.cosine_similarity(adapted, natural, dim=-1).mean()
        
        improvement = cos_sim_adapted - cos_sim
    
    return {
        'original_similarity': cos_sim.item(),
        'adapted_similarity': cos_sim_adapted.item(),
        'improvement': improvement.item()
    }


def train(model, synthetic=None, natural=None, epochs=50, batch_size=5, lr=1e-3, 
          checkpoint_path="natvox_model.pth", val_split=0.2, patience=10):
    """
    Enhanced training with validation and early stopping.
    
    Args:
        model: NATVOXAdapter model
        synthetic: Synthetic embeddings
        natural: Natural/target embeddings
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        checkpoint_path: Path to save model
        val_split: Validation split ratio (0-1)
        patience: Early stopping patience (stop if no improvement for N epochs)
    """
    
    # Load data
    if synthetic is None or natural is None:
        synthetic, natural = get_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    synthetic = synthetic.to(device)
    natural = natural.to(device)

    # Split into train/val
    n_samples = len(synthetic)
    n_val = max(1, int(n_samples * val_split))
    n_train = n_samples - n_val
    
    train_indices = torch.arange(n_train)
    val_indices = torch.arange(n_train, n_samples)
    
    train_synthetic = synthetic[train_indices]
    train_natural = natural[train_indices]
    val_synthetic = synthetic[val_indices]
    val_natural = natural[val_indices]
    
    dataset = torch.utils.data.TensorDataset(train_synthetic, train_natural)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    history = {
        'total_loss': [],
        'cosine_loss': [],
        'l2_loss': [],
        'train_sim': [],
        'val_improvement': [],
        'best_improvement': -float('inf'),
    }

    best_improvement = -float('inf')
    no_improvement_count = 0

    # Training loop
    print("=" * 70)
    print("🚀 NATVOX Adapter Training with Validation")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_cos = 0.0
        epoch_l2 = 0.0
        epoch_sim = 0.0

        for synthetic_batch, natural_batch in dataloader:

            synthetic_batch = synthetic_batch.to(device)
            natural_batch = natural_batch.to(device)

            # Normalize inputs
            synthetic_batch = F.normalize(synthetic_batch, p=2, dim=-1)
            natural_batch = F.normalize(natural_batch, p=2, dim=-1)

            # Forward
            adapted = model(synthetic_batch)

            # Loss
            loss, cosl, l2l = adapter_loss(adapted, natural_batch, synthetic_batch, model)

            optimizer.zero_grad()
            loss.backward()
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
        avg_loss = epoch_loss / n_train
        avg_cos = epoch_cos / n_train
        avg_l2 = epoch_l2 / n_train
        avg_sim = epoch_sim / n_train

        # Validation
        val_metrics = validate(model, val_synthetic, val_natural, device)
        
        history['total_loss'].append(avg_loss)
        history['cosine_loss'].append(avg_cos)
        history['l2_loss'].append(avg_l2)
        history['train_sim'].append(avg_sim)
        history['val_improvement'].append(val_metrics['improvement'])

        # Early stopping logic
        if val_metrics['improvement'] > best_improvement:
            best_improvement = val_metrics['improvement']
            no_improvement_count = 0
            history['best_improvement'] = best_improvement
            
            # Save best model
            try:
                torch.save(model.state_dict(), checkpoint_path)
            except Exception as e:
                print(f"❌ Error saving model: {e}")
        else:
            no_improvement_count += 1

        scheduler.step(val_metrics['improvement'])

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            alpha_val = model.alpha.item() if hasattr(model, 'alpha') else 0
            status = "✅" if val_metrics['improvement'] > 0 else "❌"
            print(
                f"Epoch {epoch:3d}/{epochs} {status} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train Sim: {avg_sim:.4f} | "
                f"Val Orig: {val_metrics['original_similarity']:.4f} | "
                f"Val Adapt: {val_metrics['adapted_similarity']:.4f} | "
                f"Improvement: {val_metrics['improvement']:+.4f}"
            )

        # Early stopping
        if no_improvement_count >= patience:
            print(f"\n⏸️  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print("\n" + "=" * 70)
    print(f"✅ Training complete! Best validation improvement: {best_improvement:+.6f}")
    print("=" * 70)
    
    # Reload best model
    if best_improvement > -float('inf'):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    return model, history


# Run
if __name__ == "__main__":
    from model import NATVOXAdapter

    model = NATVOXAdapter()
    trained_model, history = train(model, epochs=100)
    
    print("\n📊 Final Results:")
    if history['best_improvement'] > 0:
        print(f"✅ Model learned to IMPROVE similarity!")
        print(f"   Best improvement: {history['best_improvement']:+.6f}")
    else:
        print(f"❌ Model failed to improve similarity")
        print(f"   Best improvement: {history['best_improvement']:+.6f}")
        print(f"\n💡 Try: Adding real audio data to voice_data/my_pairs/")
