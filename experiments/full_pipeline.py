import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.decomposition import PCA

from audio_utils import get_embedding
from config import REPORTS_DIR
from dataset import get_data
from model import NATVOXAdapter
from vocalis_pipeline import text_to_wav


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def load_adapter(checkpoint_path: str) -> NATVOXAdapter:
    model = NATVOXAdapter()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_pairs(model: NATVOXAdapter, synthetic: torch.Tensor, natural: torch.Tensor):
    with torch.no_grad():
        adapted = model(synthetic)

    syn_np = synthetic.cpu().numpy()
    nat_np = natural.cpu().numpy()
    adp_np = adapted.cpu().numpy()

    before_scores = np.array([cosine_similarity(s, n) for s, n in zip(syn_np, nat_np)])
    after_scores = np.array([cosine_similarity(a, n) for a, n in zip(adp_np, nat_np)])

    metrics = {
        "pairs": int(len(before_scores)),
        "cosine_before_mean": float(before_scores.mean()),
        "cosine_after_mean": float(after_scores.mean()),
        "cosine_improvement_mean": float((after_scores - before_scores).mean()),
        "euclidean_before_mean": float(np.linalg.norm(syn_np - nat_np, axis=1).mean()),
        "euclidean_after_mean": float(np.linalg.norm(adp_np - nat_np, axis=1).mean()),
        "mse_before_mean": float(np.mean((syn_np - nat_np) ** 2)),
        "mse_after_mean": float(np.mean((adp_np - nat_np) ** 2)),
    }
    return metrics, syn_np, nat_np, adp_np, before_scores, after_scores


def run_ablation_alpha(model: NATVOXAdapter, synthetic: torch.Tensor, natural: torch.Tensor):
    original_alpha = float(model.alpha.item())
    alphas = np.linspace(0.0, 1.0, 9)
    scores = []

    with torch.no_grad():
        for alpha in alphas:
            model.alpha.data = torch.tensor(alpha, dtype=model.alpha.dtype)
            adapted = model(synthetic).cpu().numpy()
            nat = natural.cpu().numpy()
            score = np.mean([1 - cosine_distance(a, n) for a, n in zip(adapted, nat)])
            scores.append(float(score))

        model.alpha.data = torch.tensor(original_alpha, dtype=model.alpha.dtype)

    return {
        "alphas": alphas.tolist(),
        "scores": scores,
        "learned_alpha": original_alpha,
    }


def run_tts_quickcheck(model: NATVOXAdapter, target_audio: str, output_dir: Path):
    demo_texts = [
        "This is a synthetic voice for adaptation testing.",
        "Natvox adapter should improve speaker similarity.",
        "Voice conversion quality improves after adaptation.",
    ]
    modes = [("gtts_normal", False), ("gtts_slow", True)]
    rows = []

    target_emb = get_embedding(target_audio)
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode_name, slow in modes:
        for idx, text in enumerate(demo_texts):
            wav_path = output_dir / f"{mode_name}_{idx}.wav"
            text_to_wav(text, str(wav_path), slow=slow)

            syn_emb = get_embedding(str(wav_path))
            syn_t = torch.tensor(syn_emb, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                adp_emb = model(syn_t).squeeze(0).cpu().numpy()

            before = cosine_similarity(syn_emb, target_emb)
            after = cosine_similarity(adp_emb, target_emb)
            rows.append(
                {
                    "mode": mode_name,
                    "text_index": idx,
                    "before": before,
                    "after": after,
                    "improvement": after - before,
                }
            )
    return rows


def save_plots(report_dir: Path, syn_np, nat_np, adp_np, before_scores, after_scores, ablation):
    report_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.hist(before_scores, bins=15, alpha=0.6, label="Before")
    plt.hist(after_scores, bins=15, alpha=0.6, label="After")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title("Distribution: Before vs After")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "cosine_distribution.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(before_scores, after_scores, alpha=0.7)
    lims = [0, 1]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Before")
    plt.ylabel("After")
    plt.title("Per-pair Cosine: Before vs After")
    plt.tight_layout()
    plt.savefig(report_dir / "before_after_scatter.png")
    plt.close()

    all_embs = np.vstack([syn_np, nat_np, adp_np])
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(all_embs)
    n = len(syn_np)
    syn_2d = transformed[:n]
    nat_2d = transformed[n : 2 * n]
    adp_2d = transformed[2 * n :]

    plt.figure(figsize=(8, 6))
    plt.scatter(syn_2d[:, 0], syn_2d[:, 1], s=20, label="Synthetic", alpha=0.8)
    plt.scatter(nat_2d[:, 0], nat_2d[:, 1], s=20, label="Natural", alpha=0.8)
    plt.scatter(adp_2d[:, 0], adp_2d[:, 1], s=20, label="Adapted", alpha=0.8)
    plt.title("PCA Projection of Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "embedding_pca.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(ablation["alphas"], ablation["scores"], marker="o")
    plt.axvline(ablation["learned_alpha"], linestyle="--", color="red", label="Learned alpha")
    plt.xlabel("Residual alpha")
    plt.ylabel("Mean cosine similarity")
    plt.title("Ablation: Residual Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "ablation_alpha.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run full Natvox report pipeline")
    parser.add_argument("--data-dir", required=True, help="Dataset folder with synthetic/ and natural/")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint .pth")
    parser.add_argument("--output-dir", default=str(REPORTS_DIR), help="Folder to save reports")
    parser.add_argument("--target-audio", default=None, help="Optional target wav for TTS quick-check")
    args = parser.parse_args()

    report_dir = Path(args.output_dir)
    model = load_adapter(args.checkpoint)
    synthetic, natural = get_data(data_dir=args.data_dir)
    metrics, syn_np, nat_np, adp_np, before_scores, after_scores = evaluate_pairs(
        model, synthetic, natural
    )
    ablation = run_ablation_alpha(model, synthetic, natural)
    save_plots(report_dir, syn_np, nat_np, adp_np, before_scores, after_scores, ablation)

    payload = {
        "metrics": metrics,
        "ablation_alpha": ablation,
    }

    if args.target_audio:
        tts_rows = run_tts_quickcheck(model, args.target_audio, report_dir / "tts_samples")
        payload["tts_quickcheck"] = tts_rows

    with open(report_dir / "full_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(report_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Natvox Full Report\n")
        f.write("==================\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write(f"learned_alpha: {ablation['learned_alpha']}\n")
        if args.target_audio:
            f.write("tts_quickcheck: included\n")

    print(f"Saved full report to: {report_dir.resolve()}")


if __name__ == "__main__":
    main()
