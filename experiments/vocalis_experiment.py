"""
Local experiment runner adapted from Colab-style VOCALIS cells.

Usage:
  python experiments/vocalis_experiment.py --source path/to/source.wav --target path/to/target.wav

This script:
1) extracts embeddings for source/target
2) trains NATVOX adapter on fallback or paired data
3) evaluates cosine before/after
4) saves plots and JSON results to artifacts/experiment
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from audio_utils import get_embedding
from dataset import get_data
from model import NATVOXAdapter
from train import train


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def evaluate_single(model: NATVOXAdapter, source_audio: str, target_audio: str):
    src_emb = get_embedding(source_audio)
    tgt_emb = get_embedding(target_audio)

    src_tensor = torch.tensor(src_emb, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        adapted = model(src_tensor).squeeze(0).cpu().numpy()

    before = cosine(src_emb, tgt_emb)
    after = cosine(adapted, tgt_emb)
    return before, after


def main():
    parser = argparse.ArgumentParser(description="VOCALIS local experiment")
    parser.add_argument("--source", required=True, help="Path to synthetic/source audio")
    parser.add_argument("--target", required=True, help="Path to natural target audio")
    parser.add_argument("--data-dir", default=None, help="Optional paired dataset dir")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--output-dir", default="artifacts/experiment")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic, natural = get_data(data_dir=args.data_dir)
    model = NATVOXAdapter()
    model, history = train(
        model,
        synthetic=synthetic,
        natural=natural,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_path=str(out_dir / "natvox_model.pth"),
    )

    before, after = evaluate_single(model, args.source, args.target)
    metrics = {
        "single_before_cosine": before,
        "single_after_cosine": after,
        "single_improvement": after - before,
    }

    plt.figure(figsize=(8, 4))
    plt.plot(history["total_loss"], label="total_loss")
    plt.plot(history["sim"], label="train_cosine")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png")
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.bar(["Before", "After"], [before, after], color=["#E67E22", "#27AE60"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Cosine Similarity")
    plt.title("Single Pair Similarity")
    plt.tight_layout()
    plt.savefig(out_dir / "single_pair_similarity.png")
    plt.close()

    with open(out_dir / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "history": history}, f, indent=2)

    print("Experiment completed.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
