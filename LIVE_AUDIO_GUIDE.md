# Live Audio Input/Output Guide

## Overview
NATVOX-AI now supports **real-time live audio streaming** through two interfaces:
1. **Streamlit Web UI** - Interactive live inference with real-time visualizations
2. **CLI Tool** - Command-line live inference for headless/server environments

## Features
- 🎤 Real-time audio capture from your microphone
- ⚡ Streaming embedding adaptation with minimal latency
- 📊 Live metrics display (similarity scores, RMS levels)
- 📈 Real-time visualization of similarity trends
- 🔊 Audio device selection (choose input device)
- ⏱️ Session duration control
- 💾 Metrics aggregation and statistics

## Installation

First, install the updated requirements:

```bash
pip install -r requirements.txt
```

This includes the new `sounddevice` package for real-time audio I/O.

## Method 1: Streamlit Web Interface (Recommended)

### Start the App
```bash
streamlit run app.py
```

### Using Live Inference
1. Navigate to the **"Live Inference"** tab from the sidebar
2. Select your trained model checkpoint path
3. Choose your input audio device from the dropdown
4. Set buffer duration (default: 2.0 seconds)
5. Optionally set a session duration limit (0 = infinite)
6. Click **"Start Live Inference"** button

### What You'll See
- **Real-time metrics** (current similarity, average, chunks processed)
- **Live chart** showing similarity trends over time
- **Distribution plots** at the end of the session
- **Statistical summary** including min/max/std dev

### Example Workflow
```
1. Train model: Train page → Start training
2. Go to Live Inference page
3. Select model checkpoint from artifacts/
4. Select your microphone
5. Click "Start Live Inference"
6. Speak into the microphone - see metrics update in real-time
7. Let Ctrl+C or wait for session duration to stop
```

## Method 2: CLI Tool

### Basic Usage
```bash
python live_inference.py --checkpoint artifacts/natvox_model.pth
```

### List Available Devices
First, see what audio devices are available:
```bash
python live_inference.py --list-devices
```

Output example:
```
============================================================
📢 AVAILABLE INPUT DEVICES:
============================================================
  [0] Microphone (USB Audio CODEC    )
  [1] Stereo Mix (Realtek High Defin)
============================================================

🔊 AVAILABLE OUTPUT DEVICES:
============================================================
  [0] Speakers (Realtek High Defin)
  [1] HDMI Audio
============================================================
```

### Run with Specific Device
```bash
python live_inference.py \
  --checkpoint artifacts/natvox_model.pth \
  --input-device 0 \
  --output-device 0
```

### Run for Fixed Duration
```bash
python live_inference.py \
  --checkpoint artifacts/natvox_model.pth \
  --duration 30
```
This runs live inference for 30 seconds then exits.

### Full Example
```bash
python live_inference.py \
  --checkpoint artifacts/natvox_model.pth \
  --input-device 0 \
  --output-device 0 \
  --duration 60
```

## Real-Time Metrics Explanation

### Displayed Metrics

| Metric | Meaning |
|--------|---------|
| **Current Similarity** | Cosine similarity between original and adapted embedding (current chunk) |
| **Average Similarity** | Mean similarity across all chunks processed so far |
| **Chunks Processed** | Number of audio chunks analyzed |
| **Input RMS** | Root Mean Square level of audio input (volume indicator) |
| **Min/Max Similarity** | Range of similarity scores |
| **Std Dev** | Standard deviation of similarity scores |

### Interpreting Results

- **Similarity near 0.9+**: Excellent adaptation - embedding closely aligned to target manifold
- **Similarity 0.7-0.9**: Good adaptation - moderate alignment improvement
- **Similarity < 0.7**: Fair adaptation - may indicate misalignment or silence
- **High Std Dev**: Variable adaptation quality - may have different speaker characteristics

## Architecture Details

### Live Audio Pipeline
```
Microphone (sounddevice)
    ↓
Audio Buffer (2-5 seconds)
    ↓
Preprocess Audio (normalize, denoise)
    ↓
Extract Embedding (resemblyzer VoiceEncoder)
    ↓
Adapt Embedding (NATVOXAdapter model)
    ↓
Compute Metrics (cosine similarity, RMS)
    ↓
Queue Results → UI/Log Output
```

### Processing Specifications
- **Sample Rate**: 16kHz (industry standard for speech)
- **Chunk Size**: 2048 samples (~128ms at 16kHz)
- **Buffer Duration**: Configurable (0.5-5.0 seconds)
- **Latency**: ~200-500ms (embedding extraction + model inference)
- **Threading**: Non-blocking async processing

## Troubleshooting

### Issue: "No input devices found"
**Solution**: Check audio device configuration
```bash
# List devices
python live_inference.py --list-devices

# Ensure microphone is connected and recognized by OS
# Try specifying device explicitly
python live_inference.py --checkpoint model.pth --input-device 0
```

### Issue: "Checkpoint not found"
**Solution**: Verify model path
```bash
# Make sure you trained the model first
python main.py --mode train

# Or provide full path to existing checkpoint
python live_inference.py --checkpoint /path/to/natvox_model.pth
```

### Issue: Low/no audio is being captured
**Possible causes**:
- Microphone volume too low
- Wrong input device selected
- Microphone not properly connected
- OS muted/disabled the input

**Solution**:
```bash
# Try different input device
python live_inference.py --list-devices
python live_inference.py --checkpoint model.pth --input-device 1
```

### Issue: Very low similarity scores (< 0.5)
**Possible causes**:
- Model not trained properly
- Background noise affecting embedding extraction
- Silent audio input
- Speaker characteristics very different from training data

**Solution**:
- Retrain model with more/better data: `python main.py --mode train --data-dir voice_data/my_pairs`
- Ensure good microphone placement
- Reduce background noise
- Speak clearly

### Issue: Streamlit app crashes during live inference
**Solution**: 
- Ensure sounddevice is properly installed: `pip install --upgrade sounddevice`
- Try closing other audio applications
- Restart the Streamlit app

## Performance Considerations

### Hardware Requirements
- **CPU**: Multi-core recommended for smooth processing
- **GPU**: Optional, but model runs efficiently on CPU
- **RAM**: 2GB+ recommended
- **Audio Interface**: Any standard audio device (USB, built-in)

### Optimization Tips

1. **Reduce buffer duration** for lower latency (but less stable embeddings)
   ```bash
   # In Streamlit UI, set "Buffer duration" to 0.5 seconds
   ```

2. **Process in background thread** (already implemented in live_audio.py)

3. **Use headless inference** if running on server
   ```bash
   python live_inference.py --checkpoint model.pth --duration 3600
   ```

4. **Monitor CPU usage** during long sessions
   ```bash
   # On Windows PowerShell
   Get-Process | Where-Object {$_.Name -eq "python"} | Format-Table
   ```

## Advanced Usage

### Batch Processing Multiple Sessions
```bash
#!/bin/bash
# Process multiple speakers in sequence
for speaker in speaker1 speaker2 speaker3; do
    echo "Processing $speaker..."
    python live_inference.py \
        --checkpoint artifacts/natvox_model.pth \
        --duration 30 >> results_${speaker}.log
done
```

### Custom Embedding Analysis
Extend `live_audio.py` to save embeddings to file:
```python
from live_audio import LiveAudioProcessor
import numpy as np

processor = LiveAudioProcessor(model)
processor.start_capture()

# ... run inference ...

# Save all adapted embeddings
embeddings = np.array(processor.metrics['adapted_embeddings'])
np.save('adapted_embeddings.npy', embeddings)
```

### Integration with External Systems
The `LiveAudioProcessor` class can be imported and used in your own Python code:

```python
from live_audio import LiveAudioProcessor
from model import NATVOXAdapter
import torch

# Load model
model = NATVOXAdapter()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create processor
processor = LiveAudioProcessor(model)
processor.start_capture()

# Process for 30 seconds
import time
time.sleep(30)

# Get results
metrics = processor.get_metrics_summary()
print(f"Average similarity: {metrics['avg_similarity']:.4f}")

processor.stop_capture()
```

## File Structure

```
natvox-ai/
├── live_audio.py              # Core live audio module
├── live_inference.py          # CLI tool for live inference
├── app.py                      # Updated Streamlit app with Live Inference tab
├── audio_utils.py             # Audio utilities (embedding extraction)
├── model.py                   # NATVOXAdapter model
└── requirements.txt           # Updated with sounddevice
```

## API Reference

### LiveAudioProcessor

```python
class LiveAudioProcessor:
    def __init__(model, sample_rate=16000, chunk_size=2048, buffer_duration=2.0)
    def start_capture(device=None)           # Start audio stream
    def stop_capture()                        # Stop audio stream
    def get_latest_results(timeout=0.1)      # Get latest chunk results
    def get_metrics_summary()                 # Get aggregated metrics
    def reset_metrics()                       # Reset metrics for new session
```

### LiveAudioPlayback

```python
class LiveAudioPlayback:
    def __init__(sample_rate=16000, channels=1)
    def start_playback(device=None)          # Start output stream
    def play_audio(audio_data, blocking=False)  # Play audio
    def stop_playback()                      # Stop output stream
```

## Support & Contributing

For issues or feature requests related to live audio:
1. Check troubleshooting section above
2. Verify sounddevice is installed correctly
3. Test with different audio devices
4. Check system audio settings in OS

## References

- [sounddevice documentation](https://python-sounddevice.readthedocs.io/)
- [resemblyzer for embedding extraction](https://github.com/resemble-ai/resemblyzer)
- [NATVOX-AI architecture](ARCHITECTURE.md)
