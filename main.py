import argparse
import json
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from config import ARTIFACTS_DIR
from model import NATVOXAdapter
from dataset import get_data
from train import train


def evaluate_model(model, synthetic, natural):
    model.eval()
    with torch.no_grad():
        adapted = model(synthetic)

        cos = torch.nn.CosineSimilarity(dim=1)
        before = cos(synthetic, natural).mean().item()
        after = cos(adapted, natural).mean().item()
        euclid_before = torch.norm(synthetic - natural, p=2, dim=-1).mean().item()
        euclid_after = torch.norm(adapted - natural, p=2, dim=-1).mean().item()
        mse_before = torch.mean((synthetic - natural) ** 2).item()
        mse_after = torch.mean((adapted - natural) ** 2).item()

    return {
        "before_cosine": before,
        "after_cosine": after,
        "cosine_improvement": after - before,
        "before_euclidean": euclid_before,
        "after_euclidean": euclid_after,
        "before_mse": mse_before,
        "after_mse": mse_after,
    }


def save_graphs(history, metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(history["total_loss"], label="total_loss")
    plt.plot(history["cosine_loss"], label="cosine_loss")
    plt.plot(history["l2_loss"], label="l2_loss")
    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    labels = ["Before", "After"]
    values = [metrics["before_cosine"], metrics["after_cosine"]]
    colors = ["#E67E22", "#27AE60"]
    plt.bar(labels, values, color=colors)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Cosine Similarity")
    plt.title("Embedding Similarity Improvement")
    plt.tight_layout()
    plt.savefig(output_dir / "similarity_bar.png")
    plt.close()


def run_training(data_dir=None, output_dir="artifacts", epochs=50, batch_size=5):
    output_path = Path(output_dir)
    synthetic, natural = get_data(data_dir=data_dir)
    model = NATVOXAdapter()

    checkpoint_path = output_path / "natvox_model.pth"
    model, history = train(
        model,
        synthetic,
        natural,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_path=str(checkpoint_path),
    )
    metrics = evaluate_model(model, synthetic, natural)
    save_graphs(history, metrics, output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "results.txt", "w", encoding="utf-8") as f:
        f.write(f"Before Cosine: {metrics['before_cosine']:.6f}\n")
        f.write(f"After Cosine: {metrics['after_cosine']:.6f}\n")
        f.write(f"Cosine Improvement: {metrics['cosine_improvement']:.6f}\n")
        f.write(f"Before Euclidean: {metrics['before_euclidean']:.6f}\n")
        f.write(f"After Euclidean: {metrics['after_euclidean']:.6f}\n")
        f.write(f"Before MSE: {metrics['before_mse']:.6f}\n")
        f.write(f"After MSE: {metrics['after_mse']:.6f}\n")

    with open(output_path / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "history": history,
                "data_dir": data_dir or "synthetic_fallback",
                "epochs": epochs,
                "batch_size": batch_size,
            },
            f,
            indent=2,
        )

    print("\nResults:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"\nSaved outputs in: {output_path.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate NATVOX adapter')
    parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='Mode: train (default) or eval')
    parser.add_argument('--data-dir', default=None, help='Path containing synthetic/ and natural/ paired audio folders')
    parser.add_argument('--output-dir', default=str(ARTIFACTS_DIR), help='Output directory for results and graphs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size')
    args = parser.parse_args()

    if args.mode == 'train':
        run_training(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        print('To evaluate a saved model, use inference.py with --input and --target paths.')
