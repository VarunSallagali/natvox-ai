import argparse
from pathlib import Path


DEFAULT_PATTERNS = [
    "temp*.wav",
    "temp*.mp3",
    "converted.wav",
    "adapted_output.wav",
    "sample.wav",
]


def cleanup(root: Path, dry_run: bool):
    removed = []
    for pattern in DEFAULT_PATTERNS:
        for p in root.glob(pattern):
            if p.is_file():
                removed.append(p)
                if not dry_run:
                    p.unlink()
    return removed


def ensure_folders(root: Path):
    wanted = [
        root / "artifacts",
        root / "artifacts" / "reports",
        root / "artifacts" / "experiment",
        root / "paper",
        root / "experiments",
        root / "tools",
        root / "voice_data" / "my_pairs" / "synthetic",
        root / "voice_data" / "my_pairs" / "natural",
    ]
    for folder in wanted:
        folder.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Clean temp files and normalize folder layout")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--dry-run", action="store_true", help="Show files without deleting")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    ensure_folders(root)
    removed = cleanup(root, args.dry_run)

    if removed:
        print("Candidate temp files:")
        for p in removed:
            print(f"- {p}")
    else:
        print("No temp files matched cleanup patterns.")

    print("Folder layout checked/created.")


if __name__ == "__main__":
    main()
