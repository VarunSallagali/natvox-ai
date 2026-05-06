from pathlib import Path


PROJECT_ROOT = Path(".")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
EXPERIMENT_DIR = ARTIFACTS_DIR / "experiment"
VOICE_DATA_DIR = PROJECT_ROOT / "voice_data" / "my_pairs"

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
