# NATVOX-AI: Neural Adapter for Synthetic-to-Natural Speech Embedding Alignment

## Overview
A lightweight neural adapter that realigns synthetic speech embeddings to the natural speech manifold for improved voice conversion. Enables high-fidelity speaker preservation in zero-shot TTS systems.

## Key Features
- **Lightweight**: 330k parameters, trains on ~30 paired examples
- **Fast**: 90-120s training on GPU
- **Real-time**: Live audio streaming with metrics
- **Multilingual**: Works with English and Portuguese
- **Web UI**: Interactive Streamlit interface

## System Requirements
- Python 3.8+
- PyTorch 1.9+
- GPU recommended (CPU supported)

## Installation

1. Clone and setup:
```bash
git clone git clone https://github.com/VarunSallagali/natvox-ai.git
cd natvox-ai
python -m venv venv310
```

2. Activate (Windows):
```bash
venv310\Scripts\activate
```
Or (macOS/Linux):
```bash
source venv310/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
**Windows note**: If you hit build issues, use `pip install -r requirements.txt --no-build-isolation`

## Quick Start

### Web Interface (Recommended)
```bash
streamlit run app.py
```
Access the interactive dashboard with training, inference, and live audio streaming.

### Live Audio Streaming
```bash
python live_inference.py --checkpoint artifacts/natvox_model.pth
```

## Usage

### Train
```bash
python main.py --mode train --output-dir artifacts
```

### Train with Your Data
Prepare voice pairs in this structure:
```
my_voice_data/
  synthetic/
    utt_001.wav
    ...
  natural/
    utt_001.wav
    ...
```

Then train:
```bash
python main.py --mode train --data-dir my_voice_data --output-dir artifacts --epochs 50 --batch-size 5
```

### Inference
```bash
python inference.py --input sample.wav --target target.wav --output-dir artifacts
```

### Advanced Pipeline (Your TTS → Text Encoder → NATVOX-AI Adapter → VITS → GAN Vocoder)
```bash
python vocalis_pipeline.py --text "Hello world" --target target.wav --output-dir artifacts
```

## Project Structure
- `main.py` - Main training script
- `model.py` - Adapter architecture
- `app.py` - Streamlit web interface
- `live_inference.py` - CLI for real-time audio
- `inference.py` - Single-file inference
- `artifacts/` - Outputs and model checkpoints

## Documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details
- [LIVE_AUDIO_GUIDE.md](LIVE_AUDIO_GUIDE.md) - Real-time audio reference
- [LIVE_AUDIO_QUICKSTART.md](LIVE_AUDIO_QUICKSTART.md) - Quick audio setup