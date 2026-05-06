Paired voice dataset structure for NATVOX-AI.

Put your files like this:

my_pairs/
  synthetic/
    utt_001.wav
    utt_002.wav
  natural/
    utt_001.wav
    utt_002.wav

Rules:
- Both `synthetic/` and `natural/` folders are required.
- File names must match by stem (`utt_001` must exist in both folders).
- Recommended at least 30 matched pairs.
- Supported formats: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`.

Train command example:
python main.py --mode train --data-dir "voice_data/my_pairs" --output-dir "artifacts" --epochs 50 --batch-size 5
