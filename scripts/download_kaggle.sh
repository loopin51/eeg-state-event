#!/usr/bin/env bash
# Download raw EEG datasets from the Kaggle mirrors used in the 1st study.
#   SEED-IV : phhasian0710/seed-iv        (raw EEG, 15 subj x 3 sessions, 62 ch, 200 Hz)
#   DEAP    : manh123df/deap-dataset      (data_preprocessed_python, 32 subj, 32 ch, 128 Hz, ~2.9 GB)
# Requires Kaggle API credentials: ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY.
# Usage: bash scripts/download_kaggle.sh [seediv|deap|all]   (default: all)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="${KAGGLE_BIN:-$ROOT/env/bin/kaggle}"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"
what="${1:-all}"

dl() {  # dl <name> <kaggle-slug>
  local name="$1" slug="$2" dest="$RAW/$1"
  if [ -f "$dest/.complete" ]; then echo "[skip] $name already complete"; return; fi
  mkdir -p "$dest"
  echo "[download] $slug -> $dest"
  "$KAGGLE" datasets download -d "$slug" -p "$dest" --unzip
  touch "$dest/.complete"
  echo "[done] $name: $(du -sh "$dest" | cut -f1)"
}

case "$what" in
  seediv) dl seediv phhasian0710/seed-iv ;;
  deap)   dl deap   manh123df/deap-dataset ;;
  all)    dl deap   manh123df/deap-dataset; dl seediv phhasian0710/seed-iv ;;
  *) echo "unknown: $what"; exit 1 ;;
esac
