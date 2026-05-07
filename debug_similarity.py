"""
Debug script to diagnose why adapted similarity is lower than original.
"""
import torch
import torch.nn.functional as F
import numpy as np
from model import NATVOXAdapter
from audio_utils import get_embedding


def test_adapter_improvement():
    """Test if the adapter actually improves similarity on a simple example."""
    
    print("=" * 60)
    print("🔍 NATVOX Adapter Diagnostic")
    print("=" * 60)
    
    # Load model
    model = NATVOXAdapter()
    try:
        state = torch.load('natvox_model.pth', map_location='cpu')
        model.load_state_dict(state)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    model.eval()
    
    # Check model parameters
    print(f"\n📊 Model Parameters:")
    print(f"  - Embedding dim: {model.embedding_dim}")
    print(f"  - Alpha (residual weight): {model.alpha.item():.6f}")
    print(f"  - Total params: {sum(p.numel() for p in model.parameters())}")
    
    # Create synthetic test case
    print(f"\n🧪 Testing with synthetic embeddings...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Synthetic embedding (random)
    syn_emb = torch.randn(1, 256)
    syn_emb = F.normalize(syn_emb, p=2, dim=-1)
    
    # Target embedding (random, but different)
    tgt_emb = torch.randn(1, 256)
    tgt_emb = F.normalize(tgt_emb, p=2, dim=-1)
    
    # Get adapted embedding
    with torch.no_grad():
        adapted = model(syn_emb)
    
    # Compute similarities
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    original_sim = cos_sim(syn_emb, tgt_emb).item()
    adapted_sim = cos_sim(adapted, tgt_emb).item()
    
    print(f"  Original similarity: {original_sim:.6f}")
    print(f"  Adapted similarity:  {adapted_sim:.6f}")
    print(f"  Improvement:         {adapted_sim - original_sim:+.6f}")
    
    if adapted_sim < original_sim:
        print(f"\n⚠️  WARNING: Adapter is making similarity WORSE on random data!")
        print(f"   (This may be expected if training data distribution is different)")
    elif adapted_sim > original_sim:
        print(f"\n✅ Adapter is improving similarity on random data!")
    else:
        print(f"\n⚠️  Adapter is not changing similarity.")
    
    # Test on training data (should show improvement)
    print(f"\n🎓 Testing on TRAINING DISTRIBUTION (from get_data())...")
    try:
        from dataset import get_data
        syn_train, nat_train = get_data()
        
        # Test on a batch
        model.eval()
        with torch.no_grad():
            adapted_train = model(syn_train[:5])
            
            cos_train_orig = F.cosine_similarity(syn_train[:5], nat_train[:5]).mean()
            cos_train_adapt = F.cosine_similarity(adapted_train, nat_train[:5]).mean()
            
        train_improvement = cos_train_adapt - cos_train_orig
        
        print(f"  Original similarity (train dist): {cos_train_orig.item():.6f}")
        print(f"  Adapted similarity (train dist):  {cos_train_adapt.item():.6f}")
        print(f"  Improvement:                      {train_improvement.item():+.6f}")
        
        if train_improvement > 0:
            print(f"\n✅ Model IS improving similarity on training distribution!")
            print(f"   This confirms the model was trained correctly.")
        else:
            print(f"\n❌ Model is not improving on training distribution.")
            print(f"   This indicates a training problem.")
    except Exception as e:
        print(f"  Could not test on training distribution: {e}")
    
    # Test on audio files if available
    print(f"\n🎵 Testing with actual audio files...")
    try:
        # Try to find test audio files
        import os
        from pathlib import Path
        
        voice_dir = Path("voice_data/my_pairs/synthetic")
        if voice_dir.exists():
            synthetic_files = list(voice_dir.glob("*.wav"))[:2]
            
            if len(synthetic_files) >= 2:
                syn_file = synthetic_files[0]
                tgt_file = Path("voice_data/my_pairs/natural") / syn_file.name
                
                if tgt_file.exists():
                    print(f"  Using files: {syn_file.name}")
                    syn_emb_real = get_embedding(str(syn_file))
                    tgt_emb_real = get_embedding(str(tgt_file))
                    
                    syn_tensor = torch.tensor(syn_emb_real, dtype=torch.float32).unsqueeze(0)
                    tgt_tensor = torch.tensor(tgt_emb_real, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        adapted_real = model(syn_tensor)
                    
                    orig_sim_real = cos_sim(syn_tensor, tgt_tensor).item()
                    adapt_sim_real = cos_sim(adapted_real, tgt_tensor).item()
                    
                    print(f"  Original similarity: {orig_sim_real:.6f}")
                    print(f"  Adapted similarity:  {adapt_sim_real:.6f}")
                    print(f"  Improvement:         {adapt_sim_real - orig_sim_real:+.6f}")
                    
                    if adapt_sim_real < orig_sim_real:
                        print(f"\n❌ Model is degrading real audio similarity!")
            else:
                print("  Not enough audio files for testing")
        else:
            print("  voice_data directory not found")
    except Exception as e:
        print(f"  Could not test with real audio: {e}")
    
    # Check if model weights have reasonable values
    print(f"\n🔬 Model Weight Analysis:")
    for name, param in model.named_parameters():
        shape_str = str(tuple(param.shape))
        print(f"  {name:15} | shape: {shape_str:20} | "
              f"mean: {param.mean().item():+.6f} | std: {param.std().item():.6f} | "
              f"min: {param.min().item():+.6f} | max: {param.max().item():+.6f}")


if __name__ == "__main__":
    test_adapter_improvement()
    
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS:")
    print("=" * 60)
    print("If Adapted similarity < Original similarity:")
    print("  1. Retrain the model with: python main.py --mode train")
    print("  2. Check that training data pairs are correct (synthetic ≠ natural)")
    print("  3. Try higher learning rate or more epochs")
    print("  4. Check if alpha became too large (should be ~0.1-0.5)")
    print("\nIf weights look strange (all zeros, NaN, etc):")
    print("  1. Delete natvox_model.pth")
    print("  2. Retrain from scratch")
    print("=" * 60)
