import torch
import matplotlib.pyplot as plt
from model import NATVOXAdapter
from dataset import get_data
from train import train

# Load data
synthetic, natural = get_data()

# Initialize model
model = NATVOXAdapter()

# Train
model, loss_history = train(model, synthetic, natural)

# Evaluation
try:
    model.eval()
    with torch.no_grad():
        adapted = model(synthetic)
        
        cos = torch.nn.CosineSimilarity(dim=1)
        
        before = cos(synthetic, natural).mean().item()
        after = cos(adapted, natural).mean().item()
    
    with open("results.txt", "w") as f:
        f.write(f"Before: {before}\n")
        f.write(f"After: {after}\n")
        f.write(f"Improvement: {after - before}\n")
except Exception as e:
    print(f"Error during evaluation: {e}")
    exit(1)

print("\nSimilarity Before:", before)
print("Similarity After:", after)
print("Improvement:", after - before)

# Plot loss graph
plt.plot(loss_history)
plt.savefig("loss_curve.png")
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.show()  # Removed for non-interactive environments