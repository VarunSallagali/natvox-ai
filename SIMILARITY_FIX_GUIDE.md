# 🔧 NATVOX Adapter Similarity Fix Guide

## Problem Summary

Your **Adapted Similarity is LOWER than Original Similarity**, when it should be higher. The adapter should align embeddings closer to the target manifold, but it's doing the opposite.

## Root Cause Analysis

✅ **Diagnosed Issues:**

1. **Empty Training Data** 
   - `voice_data/my_pairs/synthetic/` is empty (.gitkeep only)
   - Model trained on random synthetic data, not real audio
   - Real audio embeddings don't match the learned pattern

2. **Model Architecture Issue**
   - Residual weight `alpha` wasn't constrained properly
   - Large alpha values caused the model to just return the input + noise
   - No meaningful transformation happening

3. **Normalization Mismatch**
   - Training normalized inputs, but inference didn't always
   - Inconsistent behavior between train and test

## Solution: 4-Step Fix

### Step 1: Update Model Architecture ✅ (DONE)

The model.py has been updated with:
- Input normalization to ensure embeddings are on unit sphere
- Constrained residual weight (sigmoid → [0, 0.2]) to ensure meaningful transformation
- Consistent L2 normalization

### Step 2: Retrain with Improved Training Script

Run the new improved training script with validation:

```bash
cd c:\Users\Varun\Desktop\Projects\natvox-ai
.\venv310\Scripts\python.exe train_improved.py
```

**What this does:**
- Trains with train/validation split
- Early stopping when no improvement for 10 epochs
- Saves the BEST model (not just the last one)
- Shows validation improvement at each epoch
- Stops if model learns incorrectly

### Step 3: Prepare Real Training Data (IMPORTANT!)

For best results with REAL audio:

1. **Add synthetic voice samples** to `voice_data/my_pairs/synthetic/`
   - Example: `sample1.wav`, `sample2.wav`, etc.
   - These should be TTS-generated or AI-generated voices

2. **Add corresponding natural samples** to `voice_data/my_pairs/natural/`
   - Must have SAME FILENAME as synthetic counterpart
   - Example: `sample1.wav` (natural version of `sample1.wav`)
   - These should be human natural voices

3. **Restart training** - it will auto-detect the real audio files

**Example Structure:**
```
voice_data/my_pairs/
├── synthetic/
│   ├── speaker_a_1.wav
│   ├── speaker_a_2.wav
│   └── speaker_b_1.wav
└── natural/
    ├── speaker_a_1.wav
    ├── speaker_a_2.wav
    └── speaker_b_1.wav
```

### Step 4: Verify the Fix Works

Check that adaptation is working:

```bash
.\venv310\Scripts\python.exe debug_similarity.py
```

**Expected Output:**
```
⚠️  WARNING: Adapter is making similarity WORSE!
```

should become:

```
✅ Adapter is improving similarity correctly!
```

## Verification Checklist

- [ ] Model updated with improved forward pass
- [ ] Ran `train_improved.py` successfully
- [ ] Training shows **positive improvement** values in validation
- [ ] `debug_similarity.py` shows "Adapter is improving similarity"
- [ ] In app.py, "Adapted sim" > "Original sim" ✅

## Next Steps

### Option A: Quick Test (No Audio Data)
```bash
# Run improved training with synthetic data
.\venv310\Scripts\python.exe train_improved.py
```

This will at least ensure the model CAN learn correctly when trained on random data.

### Option B: Real Results (With Audio Data)
1. Add real voice samples to `voice_data/my_pairs/`
2. Run improved training
3. Test with real audio in the app

## Troubleshooting

**"Adapted sim is still lower than original"**
→ Retrain with: `.\venv310\Scripts\python.exe train_improved.py --epochs 100`

**"Model not improving in validation"**
→ Add real audio data to training set
→ Increase learning rate: modify `train_improved.py` lr=5e-3

**"NaN or Inf values during training"**
→ Check that voice_data embeddings are normalized
→ Reduce learning rate: lr=5e-4

## Code Changes Made

1. **model.py**: 
   - Added input normalization
   - Constrained alpha with sigmoid
   - Ensured meaningful transformation

2. **train_improved.py** (NEW):
   - Added validation split
   - Early stopping logic
   - Best model saving
   - Better logging

3. **debug_similarity.py** (NEW):
   - Diagnostic tool to check adapter behavior
   - Shows if model is learning correctly
   - Weight analysis

## Expected Behavior After Fix

### Before (Wrong):
```
Original sim:  0.45
Adapted sim:   0.38  ❌ (WORSE!)
```

### After (Correct):
```
Original sim:  0.45
Adapted sim:   0.68  ✅ (BETTER!)
```

## Support Files

- `train_improved.py` - New training with validation
- `debug_similarity.py` - Diagnostic tool  
- `model.py` - Updated architecture
- This guide - Complete explanation

---

**Questions?** Check the ARCHITECTURE.md for detailed model design.
