"""
Quick test script to verify NATVOX-AI voice conversion installation
Run this to ensure everything is properly installed before launching the web app
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    modules_to_test = [
        ('torch', 'PyTorch'),
        ('librosa', 'librosa'),
        ('soundfile', 'soundfile'),
        ('numpy', 'NumPy'),
        ('streamlit', 'Streamlit'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('sounddevice', 'sounddevice'),
    ]
    
    all_ok = True
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
            all_ok = False
    
    return all_ok


def test_local_modules():
    """Test that local modules exist and can be imported"""
    print("\n🔍 Testing local modules...")
    
    local_modules = [
        'voice_conversion',
        'viz_utils',
        'live_audio',
        'audio_utils',
        'model',
        'config',
    ]
    
    all_ok = True
    for module_name in local_modules:
        module_path = Path(f"{module_name}.py")
        if module_path.exists():
            try:
                __import__(module_name)
                print(f"✅ {module_name}.py")
            except Exception as e:
                print(f"⚠️ {module_name}.py exists but import failed: {e}")
                all_ok = False
        else:
            print(f"❌ {module_name}.py not found")
            all_ok = False
    
    return all_ok


def test_voice_conversion():
    """Test voice conversion functionality"""
    print("\n🔍 Testing voice conversion module...")
    
    try:
        from voice_conversion import get_preset_list, get_preset_description, VOICE_PRESETS
        
        presets = get_preset_list()
        print(f"✅ Found {len(presets)} voice presets:")
        for preset in presets:
            desc = get_preset_description(preset)
            print(f"   • {preset}: {desc}")
        
        print(f"✅ VOICE_PRESETS dictionary has {len(VOICE_PRESETS)} entries")
        
        return True
    except Exception as e:
        print(f"❌ Voice conversion test failed: {e}")
        return False


def test_visualization():
    """Test visualization utilities"""
    print("\n🔍 Testing visualization module...")
    
    try:
        from viz_utils import (
            plot_waveform_comparison,
            plot_spectrogram_comparison,
            plot_frequency_response,
            create_metric_cards
        )
        print("✅ All visualization functions imported successfully")
        return True
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False


def test_file_structure():
    """Test that required files exist"""
    print("\n🔍 Checking file structure...")
    
    required_files = [
        'app.py',
        'voice_conversion.py',
        'viz_utils.py',
        'live_audio.py',
        'audio_utils.py',
        'model.py',
        'config.py',
        'requirements.txt',
        'Readme.md',
        'UPGRADE_GUIDE.md',
    ]
    
    all_ok = True
    for filename in required_files:
        file_path = Path(filename)
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"✅ {filename} ({size_kb:.1f} KB)")
        else:
            print(f"❌ {filename} - NOT FOUND")
            all_ok = False
    
    return all_ok


def test_audio_processing():
    """Test basic audio processing"""
    print("\n🔍 Testing audio processing...")
    
    try:
        import numpy as np
        import librosa
        
        # Create test audio
        sr = 16000
        duration = 1.0
        test_audio = np.sin(2 * np.pi * 440 * np.arange(sr * duration) / sr).astype(np.float32)
        
        print(f"✅ Created test audio: {len(test_audio)} samples at {sr}Hz")
        
        # Test basic librosa functions
        S = librosa.feature.melspectrogram(y=test_audio, sr=sr)
        print(f"✅ Computed spectrogram: shape {S.shape}")
        
        # Test voice conversion
        from voice_conversion import apply_preset
        
        converted, metrics = apply_preset(test_audio, sr, "Natural Clean")
        print(f"✅ Applied voice preset: output shape {converted.shape}")
        print(f"   Metrics: {metrics['preset']}, pitch shift: {metrics['pitch_shift_semitones']} semitones")
        
        return True
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🎙️  NATVOX-AI Voice Conversion Installation Test")
    print("=" * 60)
    
    results = {
        "Python Imports": test_imports(),
        "Local Modules": test_local_modules(),
        "File Structure": test_file_structure(),
        "Voice Conversion": test_voice_conversion(),
        "Visualization": test_visualization(),
        "Audio Processing": test_audio_processing(),
    }
    
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to launch Streamlit app:")
        print("\n   streamlit run app.py\n")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("   Make sure all packages are installed: pip install -r requirements.txt\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
