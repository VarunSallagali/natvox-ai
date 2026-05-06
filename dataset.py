from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from audio_utils import get_embedding


SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


def _load_embedding_cache(cache_path):
    cache = np.load(cache_path)
    synthetic = torch.tensor(cache["synthetic"], dtype=torch.float32)
    natural = torch.tensor(cache["natural"], dtype=torch.float32)
    return synthetic, natural


def _save_embedding_cache(cache_path, synthetic, natural):
    np.savez(
        cache_path,
        synthetic=synthetic.detach().cpu().numpy(),
        natural=natural.detach().cpu().numpy(),
    )


def _get_paired_audio_paths(data_dir):
    data_root = Path(data_dir)
    synthetic_dir = data_root / "synthetic"
    natural_dir = data_root / "natural"

    if not synthetic_dir.exists() or not natural_dir.exists():
        raise FileNotFoundError(
            f"Missing dataset folders. Expected: {synthetic_dir} and {natural_dir}"
        )

    synthetic_paths = {
        p.stem: p
        for p in synthetic_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    }
    natural_paths = {
        p.stem: p
        for p in natural_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    }

    shared_keys = sorted(set(synthetic_paths) & set(natural_paths))
    if not shared_keys:
        raise ValueError(
            "No matching audio file names found between synthetic/ and natural/ folders."
        )

    pairs = [(synthetic_paths[k], natural_paths[k]) for k in shared_keys]
    return pairs


def load_paired_embeddings_from_audio(data_dir, use_cache=True):
    data_root = Path(data_dir)
    cache_path = data_root / "embedding_cache.npz"

    if use_cache and cache_path.exists():
        return _load_embedding_cache(cache_path)

    pairs = _get_paired_audio_paths(data_root)
    synthetic_embeddings = []
    natural_embeddings = []

    for synthetic_path, natural_path in pairs:
        synthetic_embeddings.append(get_embedding(str(synthetic_path)))
        natural_embeddings.append(get_embedding(str(natural_path)))

    synthetic = torch.tensor(np.array(synthetic_embeddings), dtype=torch.float32)
    natural = torch.tensor(np.array(natural_embeddings), dtype=torch.float32)

    synthetic = F.normalize(synthetic, p=2, dim=-1)
    natural = F.normalize(natural, p=2, dim=-1)

    if use_cache:
        _save_embedding_cache(cache_path, synthetic, natural)

    return synthetic, natural


def get_data(data_dir=None, sample_count=30, use_cache=True):
    if data_dir:
        return load_paired_embeddings_from_audio(data_dir, use_cache=use_cache)

    torch.manual_seed(42)
    synthetic = torch.randn(sample_count, 256)

    # Synthetic fallback for quick experimentation without audio files.
    transform = torch.randn(256, 256) * 0.5
    natural = synthetic @ transform

    synthetic = F.normalize(synthetic, p=2, dim=-1)
    natural = F.normalize(natural, p=2, dim=-1)
    return synthetic, natural