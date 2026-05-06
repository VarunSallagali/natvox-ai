# Quick Start: Live Audio Features

## 🚀 Get Started in 2 Minutes

### Prerequisites
```bash
# Install/update requirements
pip install -r requirements.txt
```

### Option 1: Web Interface (Easiest) 🌐
```bash
streamlit run app.py
# Then navigate to "Live Inference" tab
```

### Option 2: Command Line (For Servers)
```bash
python live_inference.py --checkpoint artifacts/natvox_model.pth
```

---

## Quick Examples

### Check Your Audio Devices
```bash
python live_inference.py --list-devices
```

### Record for 30 seconds
```bash
python live_inference.py --checkpoint artifacts/natvox_model.pth --duration 30
```

### Use a Specific Microphone
```bash
python live_inference.py --checkpoint artifacts/natvox_model.pth --input-device 0
```

---

## What You Get

**Real-time Metrics:**
- ✅ Current embedding similarity score
- ✅ Average similarity across the session
- ✅ Number of chunks processed
- ✅ Audio input level (RMS)
- ✅ Min/Max/Std deviation statistics

**Visualizations:**
- 📊 Live similarity trend chart (in Streamlit)
- 📈 Distribution histogram (at end of session)
- 📉 Time-series with confidence bands

---

## Full Documentation

See [LIVE_AUDIO_GUIDE.md](LIVE_AUDIO_GUIDE.md) for:
- Detailed setup instructions
- Architecture explanation
- Troubleshooting guide
- Advanced usage examples
- API reference

---

## Typical Workflow

```bash
# 1. Train your model (if not already done)
python main.py --mode train --data-dir voice_data/my_pairs

# 2. Start the web app
streamlit run app.py

# 3. Go to "Live Inference" tab
# 4. Select your model checkpoint
# 5. Click "Start Live Inference"
# 6. Speak into microphone - watch metrics update!

# OR use CLI for faster testing:
python live_inference.py --checkpoint artifacts/natvox_model.pth --duration 15
```

---

## Files Added/Modified

**New Files:**
- `live_audio.py` - Core real-time audio processing module
- `live_inference.py` - CLI tool for live inference
- `LIVE_AUDIO_GUIDE.md` - Complete documentation

**Modified Files:**
- `app.py` - Added "Live Inference" page to Streamlit UI
- `requirements.txt` - Added `sounddevice` dependency

---

## Key Features

🎤 **Real-time Capture** - Stream audio directly from your microphone
⚡ **Low Latency** - ~200-500ms processing time per chunk  
📊 **Live Metrics** - See similarity scores update in real-time
🎛️ **Device Selection** - Choose input/output audio devices
⏱️ **Flexible Duration** - Run indefinitely or for fixed time
💾 **Metrics Export** - Save statistics at session end

---

## Support

**Issue:** No audio is being captured
→ Try: `python live_inference.py --list-devices` and use `--input-device` to specify

**Issue:** Low similarity scores
→ Try: Train with more data or check microphone quality

**Issue:** Streamlit app is slow
→ Try: Reduce buffer duration to 0.5-1.0 seconds

See [LIVE_AUDIO_GUIDE.md](LIVE_AUDIO_GUIDE.md#troubleshooting) for more troubleshooting
