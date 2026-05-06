# 🎙️ NATVOX-AI Voice Conversion - Complete Walkthrough

## Getting Started

### 1. Verify Installation

Before launching, test that everything is installed:

```bash
cd c:\Users\Varun\Desktop\Projects\natvox-ai
python test_installation.py
```

Expected output:
```
✅ PyTorch
✅ librosa
✅ soundfile
... (all green checks)
🎉 All tests passed! Ready to launch Streamlit app
```

### 2. Launch the Application

```bash
streamlit run app.py
```

Browser will open automatically to `http://localhost:8501`

---

## 🎯 Complete Workflow Example

### Scenario: Convert Your Voice to "Deep Narrator"

#### Step 1: Navigate to Voice Conversion

In the left sidebar, click **"🎯 Voice Conversion"**

You'll see:
- Upload section at top
- Preset selector
- Apply button

#### Step 2: Upload Audio

Click **"Select audio file"** and choose your audio:
- `.wav` - Best quality
- `.mp3` - Common format
- `.m4a` - iPhone recordings
- `.flac`, `.ogg` - Also supported

**Example:** Upload `my_voice.wav` (5 seconds of speech)

#### Step 3: Select Preset

Click the dropdown next to **"Choose a voice preset"**

6 presets available:
1. **Natural Clean** - Professional, clear
2. **Deep Narrator** - Lower, resonant (for this example)
3. **Female Soft** - Feminine, warm
4. **Robot AI** - Synthetic, crisp
5. **Energetic Speaker** - Bright, dynamic
6. **Child Voice** - Higher, playful

**For this example:** Select "Deep Narrator"

Caption shows: *"Deep, resonant voice for narration"*

#### Step 4: View Preset Details (Optional)

Click **"View Preset Details"** expander to see:
- Pitch Shift: -5 semitones (lower pitch)
- Speed: 0.95x (slightly slower)
- Formant: -150 cents (vocal character shift)
- EQ settings, clarity, reverb, compression values

This helps you understand what the preset does.

#### Step 5: Apply Conversion

Click **"🚀 Apply Voice Conversion"** button

Status message appears: *"🔄 Converting your voice..."*

Wait 2-5 seconds (depending on audio length).

Success message: **"✅ Conversion complete!"**

#### Step 6: View Metrics

Four metric cards appear:

| Card | Value | Meaning |
|------|-------|---------|
| Original RMS | 0.1234 | Input audio power |
| Duration | 5.23s | Length of audio |
| Quality Change | +8.5% | Improvement badge |
| Pitch Shift | -5st | 5 semitones lower |

Second row of metrics:
| Card | Value | Meaning |
|------|-------|---------|
| Speed Change | -5% | 5% slower |
| Preset | Deep Narrator | Applied effect |
| Format | 16000 Hz | Sample rate |

#### Step 7: Listen to Audio

**Before & After Comparison** section shows:

**Original Audio**
- Waveform with play button
- Duration: 5.23s | Sample Rate: 16000Hz

Click ▶️ to hear your original voice

**Converted Audio**
- Waveform (might look different) with play button
- Duration: 5.23s | Sample Rate: 16000Hz

Click ▶️ to hear your transformed voice

**You should notice:**
- Lower pitch (deeper)
- Slightly slower speech
- Different vocal character
- More resonant tone

#### Step 8: Download Files

Two download buttons:

1. **📥 Download Original**
   - Saves as `original_audio.wav`
   - Your original voice
   - Full quality

2. **📥 Download Converted**
   - Saves as `converted_audio.wav`
   - Transformed voice
   - Same sample rate

Click to download both for comparison.

#### Step 9: Visual Analysis

Click tabs to explore:

**Tab 1: Waveforms**
- Red waveform = original
- Teal waveform = converted
- Shows amplitude changes over time
- Professional visualization

**Tab 2: Spectrograms**
- Original spectrum on top (colorful plot)
- Converted spectrum below
- Frequency intensity shown by color
- White = loud frequencies
- Dark = quiet frequencies

**Observations:**
- Lower frequencies brighter in Deep Narrator (due to pitch shift)
- Frequency range unchanged (same person, different character)

**Tab 3: Frequency Response**
- Graph showing frequency content
- X-axis: Hz (log scale, 20-8000 Hz)
- Y-axis: Magnitude in dB
- Red line: Original
- Teal line: Converted

**What to look for:**
- Conversion emphasizes lower frequencies
- Subtle shifts in resonance
- No extreme peaks (good)

**Tab 4: Loudness Envelope**
- Graph of audio volume over time
- Red line: Original loudness
- Teal line: Converted loudness
- Shows if compression was applied

**Typical pattern:**
- Both lines follow similar shape
- Converted might be slightly smoother (compression)

#### Step 10: Review Conversion Summary

At bottom, summary shows:

**Original Audio**
- Duration: 5.23s
- RMS Level: 0.1234 (power)
- Peak: 0.4532 (max amplitude)
- Sample Rate: 16000Hz

**Converted Audio**
- Duration: 5.23s
- RMS Level: 0.1567 (slightly higher after DSP)
- Peak: 0.4891
- Preset: Deep Narrator

---

## 🔀 Try Other Presets

Go back and try different presets on same audio:

### Preset Comparison

Upload same audio 6 times with different presets:

1. **Natural Clean**
   - Subtle changes
   - Best for slight enhancement
   - Professional, unnoticeable

2. **Deep Narrator**
   - Obvious pitch drop
   - Deeper character
   - Slower speech
   - Great for audiobooks

3. **Female Soft**
   - Higher pitch (+4 semitones)
   - Faster (+5%)
   - Warmer tone
   - Different vocal character

4. **Robot AI**
   - Crisp, synthetic
   - Slightly higher
   - Very clear
   - Good for tech content

5. **Energetic Speaker**
   - Bright, dynamic
   - Slightly faster
   - High-energy vibe
   - Great for sales

6. **Child Voice**
   - Much higher (+7 semitones)
   - Faster (+10%)
   - Playful tone
   - Character voices

---

## 📊 Understanding the Audio Metrics

### RMS (Root Mean Square)
- Measures audio power/loudness
- Higher = louder audio
- 0.0 = silent
- 1.0 = maximum (distortion)
- Typical speech: 0.1-0.2

### Peak Amplitude
- Maximum volume point
- Should be < 1.0 (no clipping)
- Good range: 0.3-0.8

### Duration
- Total audio length in seconds
- Stays same after conversion
- Useful for comparisons

### Sample Rate
- Samples per second
- 16000 Hz = 16,000 samples/sec
- CD quality = 44100 Hz
- 16000 Hz = industry standard for speech

---

## 🔬 Inference Lab (For Embeddings)

If you want to test NATVOX embedding adapter:

1. Click **"🔬 Inference Lab"** in sidebar
2. Upload synthetic reference audio
3. Upload natural target audio
4. Click "▶️ Run Adaptation"
5. See embedding similarity scores

This tests the neural adapter (different from voice presets).

---

## 🎤 Live Inference (For Real-time)

To stream from your microphone:

1. Click **"🎤 Live Inference"** in sidebar
2. Select your microphone device
3. Set buffer duration (2.0s default)
4. Click "🔴 Start Live Inference"
5. Speak into microphone
6. Watch metrics update in real-time

---

## 📊 Training (Build Custom Adapter)

To train a custom model on your voice data:

1. Prepare dataset:
   ```
   voice_data/my_pairs/
   ├── synthetic/
   │   ├── utt_001.wav
   │   ├── utt_002.wav
   │   └── ... (30+ files)
   └── natural/
       ├── utt_001.wav
       ├── utt_002.wav
       └── ... (30+ files)
   ```

2. Click **"📊 Training"** in sidebar
3. Enter dataset directory path
4. Set epochs (50 default) and batch size (5 default)
5. Click "▶️ Start Training"
6. Wait for training to complete

---

## 💾 Troubleshooting Common Issues

### Issue: Audio plays but sounds unchanged

**Solution:**
- Upload original audio without effects
- Try more extreme preset like "Child Voice"
- Check volume isn't already maxed

### Issue: Download button doesn't work

**Solution:**
- Disable browser popup blocker
- Try different browser (Chrome/Firefox)
- Check browser console for errors

### Issue: Spectrogram takes long time

**Solution:**
- Streamlit caches after first run
- Close other applications
- Spectrogram computation is normal (1-2 seconds)

### Issue: Error "Loading audio failed"

**Solution:**
- Ensure audio file is valid format
- Try converting to WAV first
- Check file isn't corrupted
- Maximum recommended: 60 seconds audio

### Issue: App crashes during conversion

**Solution:**
- Restart Streamlit: `streamlit run app.py`
- Clear browser cache: Ctrl+Shift+Delete
- Close other heavy applications
- Check Python version is 3.8+

---

## 🎯 Best Practices

### Audio Quality Tips

✅ **DO:**
- Use clear audio recordings
- Reduce background noise first
- Speak clearly and naturally
- Use 16000 Hz sample rate
- Record in quiet room
- Keep audio under 60 seconds

❌ **DON'T:**
- Use phone speaker recordings
- Record in noisy environment
- Whisper or unclear speech
- Use extremely low quality audio
- Record with extreme compression

### Preset Selection

**For Professional:**
→ Natural Clean, Deep Narrator, Female Soft

**For Entertainment:**
→ Robot AI, Energetic Speaker, Child Voice

**For Testing:**
→ Try all 6, pick favorites

**For Production:**
→ Train custom adapter after testing presets

---

## 📈 Performance Expectations

| Task | Time | Notes |
|------|------|-------|
| Audio Upload | <1s | Instant in most cases |
| Voice Conversion | 2-5s | Depends on length |
| Waveform Plot | <1s | Fast |
| Spectrogram | 1-2s | First render slower |
| Frequency Response | <1s | Usually fast |
| Loudness Envelope | <1s | Very quick |
| Full Analysis | ~5-10s | Total for all visuals |

---

## 🚀 Advanced Usage

### Batch Processing

Process multiple files (command line):

```python
import librosa
from voice_conversion import apply_preset

preset = "Deep Narrator"
files = ["audio1.wav", "audio2.wav", "audio3.wav"]

for filename in files:
    audio, sr = librosa.load(filename, sr=16000)
    converted, metrics = apply_preset(audio, sr, preset)
    
    # Save
    import soundfile as sf
    output_name = filename.replace('.wav', '_converted.wav')
    sf.write(output_name, converted, sr)
    print(f"✅ {output_name}")
```

### Custom Presets

Add to `voice_conversion.py`:

```python
"My Preset": {
    "pitch_shift": 1,
    "speed_factor": 1.02,
    "formant_shift": 25,
    "eq_low": 1.1,
    "eq_mid": 1.15,
    "eq_high": 1.2,
    "clarity": 0.25,
    "reverb": 0.1,
    "compression": 0.35,
    "description": "Subtle enhancement"
}
```

Then reload Streamlit app.

### Programmatic API

```python
from voice_conversion import apply_preset, get_preset_list
import librosa

# Load audio
audio, sr = librosa.load('voice.wav', sr=16000)

# List available presets
print(get_preset_list())
# Output: ['Natural Clean', 'Deep Narrator', ...]

# Apply preset
converted, metrics = apply_preset(audio, sr, "Deep Narrator")

# Access metrics
print(metrics)
# {
#   'preset': 'Deep Narrator',
#   'pitch_shift_semitones': -5,
#   'speed_factor': 0.95,
#   'original_rms': 0.123,
#   'output_rms': 0.145,
#   'description': 'Deep, resonant voice for narration'
# }
```

---

## ✅ Verification Checklist

Before sharing with others:

- [ ] Upload audio works
- [ ] All 6 presets appear in dropdown
- [ ] Voice conversion runs without errors
- [ ] Can hear difference between presets
- [ ] Download buttons save audio files
- [ ] Waveform plots render
- [ ] Spectrogram shows correctly
- [ ] Frequency response graph visible
- [ ] Loudness envelope displays
- [ ] Metrics cards show values
- [ ] Original audio plays correctly
- [ ] Converted audio plays correctly
- [ ] Training still works
- [ ] Live Inference still works
- [ ] No performance issues

---

## 🎉 You're Ready!

You now have a **professional voice conversion studio** integrated into your NATVOX-AI project.

### What You Can Do:

✅ Convert voices with 6 professional presets  
✅ Analyze before/after with visualizations  
✅ Download transformed audio  
✅ Train custom embedding adapters  
✅ Test with real-time microphone input  
✅ Maintain all existing NATVOX features  

**Enjoy! 🎙️**

---

## 📞 Need Help?

See these files:
- `UPGRADE_GUIDE.md` - Technical details
- `LIVE_AUDIO_GUIDE.md` - Real-time audio
- `voice_conversion.py` - Code comments
- `viz_utils.py` - Visualization code

Good luck with your voice conversion studio! 🚀
