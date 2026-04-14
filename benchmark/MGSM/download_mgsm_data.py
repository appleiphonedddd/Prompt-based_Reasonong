#!/usr/bin/env python3
"""
Download MGSM dataset TSV files directly from HuggingFace.
"""

import subprocess
from pathlib import Path

# Create data directory
DATA_DIR = Path("benchmark/MGSM/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "th": "Thai",
    "sw": "Swahili",
    "bn": "Bengali",
}

BASE_URL = "https://huggingface.co/datasets/juletxara/mgsm/raw/main"

print("Downloading MGSM dataset TSV files...")
print(f"Saving to: {DATA_DIR.absolute()}")
print()

for lang_code, lang_name in LANGUAGES.items():
    file_url = f"{BASE_URL}/mgsm_{lang_code}.tsv"
    dest_file = DATA_DIR / f"mgsm_{lang_code}.tsv"

    print(f"Downloading {lang_name:12} ({lang_code})...", end=" ", flush=True)

    result = subprocess.run(
        ["curl", "-sL", "-o", str(dest_file), file_url],
        capture_output=True,
        timeout=30,
    )

    if result.returncode == 0 and dest_file.exists():
        size_kb = dest_file.stat().st_size / 1024
        print(f"✓ ({size_kb:.1f} KB)")
    else:
        print(f"✗ Failed")

print()
print("Download complete!")

# List files
files = sorted(DATA_DIR.glob("*.tsv"))
if files:
    print(f"\nDownloaded {len(files)} files:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name:20} ({size_kb:7.1f} KB)")
