import argparse
from pathlib import Path


SUPPORTED = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def scan(folder: Path):
    return {p.stem: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED}


def main():
    parser = argparse.ArgumentParser(description="Validate paired synthetic/natural dataset")
    parser.add_argument("--data-dir", required=True, help="Dataset root containing synthetic/ and natural/")
    args = parser.parse_args()

    root = Path(args.data_dir)
    syn = root / "synthetic"
    nat = root / "natural"
    if not syn.exists() or not nat.exists():
        raise FileNotFoundError(f"Expected folders: {syn} and {nat}")

    syn_map = scan(syn)
    nat_map = scan(nat)
    shared = sorted(set(syn_map) & set(nat_map))
    only_syn = sorted(set(syn_map) - set(nat_map))
    only_nat = sorted(set(nat_map) - set(syn_map))

    print(f"synthetic files: {len(syn_map)}")
    print(f"natural files:   {len(nat_map)}")
    print(f"paired files:    {len(shared)}")

    if only_syn:
        print("\nMissing in natural/:")
        for k in only_syn[:20]:
            print(f"- {k}")
    if only_nat:
        print("\nMissing in synthetic/:")
        for k in only_nat[:20]:
            print(f"- {k}")

    if len(shared) < 30:
        print("\nWarning: fewer than 30 pairs. Recommended minimum is 30.")
    else:
        print("\nDataset pair count looks good.")


if __name__ == "__main__":
    main()
