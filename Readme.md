# NATVOX-AI: Neural Adapter for Synthetic-to-Natural Speech Embedding Alignment

## Overview
NATVOX-AI is a compact three-layer neural adapter designed to realign synthetic speaker embeddings with the natural speech manifold. It addresses the distributional mismatch between TTS-generated and natural speech embeddings in zero-shot voice conversion systems, enabling high-fidelity speaker identity preservation when using synthetic references.

## Key Features
- **Lightweight Architecture**: 330k parameters, trained on 30 paired examples
- **Bottleneck Design**: 256 → 512 → 256 → 256 with residual weighting
- **Domain Adaptation**: Maps synthetic embeddings to natural manifold
- **Integration**: Slots directly into YourTTS conditioning pipeline
- **Multilingual Support**: Effective on English and Portuguese corpora
- **Objective Metrics**: Cosine similarity, Euclidean distance, MSE
- **Perceptual Evaluation**: MOS scores for naturalness and speaker similarity
- **🎤 Live Audio I/O**: Real-time audio streaming with live metrics display (NEW)

## Live Audio Features ✨

NATVOX-AI now supports **real-time audio input and output**:

### Quick Start
```bash
# Web interface (recommended)
streamlit run app.py
# → Go to "Live Inference" tab

# Or command-line
python live_inference.py --checkpoint artifacts/natvox_model.pth
```

### What You Get
- Stream audio from microphone in real-time
- See embedding adaptation metrics update live
- Live visualization of similarity trends
- Session statistics and distributions

📚 **Full Documentation**: See [LIVE_AUDIO_QUICKSTART.md](LIVE_AUDIO_QUICKSTART.md) and [LIVE_AUDIO_GUIDE.md](LIVE_AUDIO_GUIDE.md)

## Architecture Details
The adapter implements a feed-forward network with:
- Layer 1: Linear(256,512) → ReLU → Dropout(0.1) → LayerNorm
- Layer 2: Linear(512,256) → ReLU → Dropout(0.1) → LayerNorm  
- Layer 3: Linear(256,256)
- Residual: Output + α × Input (α learnable, init 0.5)
- Final: L2 normalization

## Training
- Loss: Cosine distance + λ₁ × Euclidean distance + λ₂ × L2 regularization
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau
- Epochs: 50, Batch size: 5
- Hardware: NVIDIA V100 (90-120s training time)

## Results
- **VCTK**: SECS 0.491 → 0.862 (75.5% improvement)
- **LibriTTS**: SECS 0.520 → 0.843 (58.7% improvement)  
- **MLS-PT**: SECS 0.498 → 0.817 (57.7% improvement)
- **MOS Naturalness**: 2.9 → 3.9 (+1.0)
- **MOS Speaker Similarity**: 2.7 → 4.2 (+1.5)
- **A/B Preference**: 78.5% prefer Natvox-AI

## Demo Readiness
For a final review, use the following commands to show the model and outputs clearly:
- `streamlit run app.py` — open the interactive web demo with live audio, metrics, and charts.
- `python live_inference.py --checkpoint artifacts/natvox_model.pth` — run real-time microphone inference from the CLI.
- `python inference.py --input sample.wav --target target.wav --output-dir artifacts` — produce one-shot evaluation files.

Expected output files for demonstration:
- `artifacts/natvox_model.pth`
- `artifacts/results.json`
- `artifacts/results.txt`
- `artifacts/loss_curve.png`
- `artifacts/similarity_bar.png`
- `artifacts/reports/full_report.json`
- `artifacts/reports/summary.txt`

## System Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended for training)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd natvox-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv310
```

On Windows:
```bash
venv310\Scripts\activate
```
On macOS / Linux:
```bash
source venv310/bin/activate
```

3. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

**Note:** If you encounter Device Guard or Visual C++ build tool issues on Windows, use:
```bash
python -m pip install -r requirements.txt --no-build-isolation
```

## Usage

### Training
```bash
python main.py --mode train --output-dir "artifacts"
```
This will train the model and save results to `artifacts/natvox_model.pth`.

### Training with real paired voice data
Prepare a directory with paired file names:
```text
my_voice_data/
  synthetic/
    utt_001.wav
    utt_002.wav
  natural/
    utt_001.wav
    utt_002.wav
```

Then run:
```bash
python main.py --mode train --data-dir "my_voice_data" --output-dir "artifacts" --epochs 50 --batch-size 5
```

Generated outputs:
- `artifacts/natvox_model.pth`
- `artifacts/results.json`
- `artifacts/results.txt`
- `artifacts/loss_curve.png`
- `artifacts/similarity_bar.png`

### Inference
```bash
python inference.py --input sample.wav --target target.wav --output-dir "artifacts"
```

### Live Audio Inference 🎤
Stream audio from your microphone in real-time:

**Web Interface:**
```bash
streamlit run app.py
# Navigate to "Live Inference" tab
```

**Command Line:**
```bash
python live_inference.py --checkpoint artifacts/natvox_model.pth

# List available audio devices
python live_inference.py --list-devices

# Run for specific duration (seconds)
python live_inference.py --checkpoint artifacts/natvox_model.pth --duration 30

# Use specific input device
python live_inference.py --checkpoint artifacts/natvox_model.pth --input-device 0
```

See [LIVE_AUDIO_GUIDE.md](LIVE_AUDIO_GUIDE.md) for detailed live audio documentation.

### Advanced pipeline
```bash
python vocalis_pipeline.py --text "Hello world" --target target.wav --output-dir "artifacts"
```

This script runs the VOCALIS adapter pipeline from text > TTS > embeddings adaptation > voice conversion DSP > similarity report. Generates enhanced synthetic audio and DSP-converted audio for comparison. If target.wav doesn't exist, it generates a sample target voice. Results saved to `artifacts/`.
    
### Unit tests
```bash
python -m pip install pytest
python -m pytest test_adapter.py -q
```

### Model format
- Saved checkpoint: `natvox_model.pth`
- Adapter architecture in `model.py`

### Additional command
- `python main.py --mode eval` prints guidance for inference pipeline

### Existing results
Output similarity improvement is stored in `results.txt`.

### Local VOCALIS-style experiment
You can run a local experiment adapted from notebook cells:
```bash
python experiments/vocalis_experiment.py --source "path/to/synthetic.wav" --target "path/to/natural.wav"
```
Optional paired-data training:
```bash
python experiments/vocalis_experiment.py --source "path/to/synthetic.wav" --target "path/to/natural.wav" --data-dir "voice_data/my_pairs"
```
Outputs are saved to `artifacts/experiment/`:
- `natvox_model.pth`
- `training_curves.png`
- `single_pair_similarity.png`
- `experiment_results.json`

### Full report pipeline (graphs + ablations + JSON summary)
Run this after training:
```bash
python experiments/full_pipeline.py --data-dir "voice_data/my_pairs" --checkpoint "artifacts/natvox_model.pth" --output-dir "artifacts/reports"
```
Optional TTS quick check:
```bash
python experiments/full_pipeline.py --data-dir "voice_data/my_pairs" --checkpoint "artifacts/natvox_model.pth" --output-dir "artifacts/reports" --target-audio "path/to/target.wav"
```
This saves:
- `artifacts/reports/full_report.json`
- `artifacts/reports/summary.txt`
- `artifacts/reports/cosine_distribution.png`
- `artifacts/reports/before_after_scatter.png`
- `artifacts/reports/embedding_pca.png`
- `artifacts/reports/ablation_alpha.png`

### Dataset and project utilities
Validate dataset pairing:
```bash
python tools/validate_dataset.py --data-dir "voice_data/my_pairs"
```

Cleanup temporary files and normalize folder layout:
```bash
python tools/cleanup_project.py --root "."
```

### Paper figure
LaTeX architecture figure is available at:
- `paper/fig_vocalis_arch.tex`

### Web Interface
```bash
streamlit run app.py
```
Open the provided URL in your browser to use the interactive web interface.

Features:
- Dataset setup guide
- One-click training with live logs
- Results dashboard with metrics and graphs
- Inference lab for before/after similarity comparison
- **Live Inference tab** for real-time audio streaming

## Project Structure
- `main.py`: Main training script
- `model.py`: NATVOXAdapter model architecture
- `train.py`: Training utilities
- `inference.py`: Inference script
- `live_inference.py`: CLI tool for real-time audio inference
- `live_audio.py`: Real-time audio processing module
- `app.py`: Streamlit web interface
- `audio_utils.py`: Audio processing utilities
- `dataset.py`: Data loading and generation
- `utils.py`: Loss functions and utilities
- `ARCHITECTURE.md`: Full architecture details

## Documentation

- **[LIVE_AUDIO_QUICKSTART.md](LIVE_AUDIO_QUICKSTART.md)**: Quick reference for real-time audio features
- **[LIVE_AUDIO_GUIDE.md](LIVE_AUDIO_GUIDE.md)**: Complete guide with troubleshooting and API reference
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Technical architecture overview