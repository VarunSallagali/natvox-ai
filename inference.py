import torch
import numpy as np
import os
from model import NATVOXAdapter
from audio_utils import get_embedding


def evaluate_audio(audio_path, target_audio_path=None, checkpoint_path="natvox_model.pth"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = NATVOXAdapter()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    synthetic_emb = get_embedding(audio_path)
    synthetic_tensor = torch.tensor(synthetic_emb, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        adapted_tensor = model(synthetic_tensor)

    results = {
        'synthetic_shape': synthetic_tensor.shape,
        'adapted_shape': adapted_tensor.shape,
        'adapted_embedding': adapted_tensor.squeeze(0).cpu().numpy()
    }

    if target_audio_path:
        target_emb = get_embedding(target_audio_path)
        target_tensor = torch.tensor(target_emb, dtype=torch.float32).unsqueeze(0)

        cos = torch.nn.CosineSimilarity(dim=1)
        results['original_similarity'] = cos(synthetic_tensor, target_tensor).item()
        results['adapted_similarity'] = cos(adapted_tensor, target_tensor).item()
        results['improvement'] = results['adapted_similarity'] - results['original_similarity']

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NATVOX inference & evaluation')
    parser.add_argument('--input', default='sample.wav', help='Path to input audio file')
    parser.add_argument('--target', default=None, help='Path to target natural audio for similarity evaluation (optional)')
    parser.add_argument('--checkpoint', default='natvox_model.pth', help='Adapter checkpoint path')

    args = parser.parse_args()
    out = evaluate_audio(args.input, args.target, args.checkpoint)

    print('Evaluation results:')
    for k, v in out.items():
        if isinstance(v, (np.ndarray, list)):
            print(f"{k}: shape={np.array(v).shape}")
        else:
            print(f"{k}: {v}")
