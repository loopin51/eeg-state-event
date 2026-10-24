#!/usr/bin/env bash
# Additional Kaggle downloads (2026-09-21): FACED, BCI2020 Track 3, SEED-VII preprocessed EEG + continuous labels.
#   FACED    : saifkhancse/faced-eeg                                  (123 subj, 32 ch, 28 clips, processed 250 Hz)
#   BCI2020  : abdulkareembageri/imagined-speech-eeg-signal-bci2020   (imagined speech, 5 phrases)
#   SEED-VII : blackfox20092006/seed-series-multimodal-eeg-eye-full   (only seed_vii/, 20 subj, 62 ch, 80 clips)
# NOTE: these are unofficial mirrors. SEED-family and FACED have their own licence terms (BCMI / Synapse);
#       obtain official access before publishing results.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; K="${KAGGLE_BIN:-/Users/loopin51/miniforge3/bin/kaggle}"; RAW="$ROOT/data/raw"
full() { local name="$1" slug="$2" dest="$RAW/$1"; [ -f "$dest/.complete" ] && { echo "[skip] $name"; return; }
  mkdir -p "$dest"; echo "[download] $slug"; "$K" datasets download -d "$slug" -p "$dest" --unzip -q && touch "$dest/.complete"; echo "[done] $name $(du -sh "$dest" | cut -f1)"; }
full faced saifkhancse/faced-eeg
full bci2020 abdulkareembageri/imagined-speech-eeg-signal-bci2020
DEST="$RAW/seedvii"; mkdir -p "$DEST"
if [ ! -f "$DEST/.complete" ]; then
  fail=0
  while read -r f; do
    out="$DEST/$f"; [ -s "$out" ] && continue
    mkdir -p "$(dirname "$out")"
    "$K" datasets download -d blackfox20092006/seed-series-multimodal-eeg-eye-full -f "$f" -p "$(dirname "$out")" -q || { echo "[fail] $f"; fail=1; continue; }
    z="$(dirname "$out")/$(basename "$f").zip"; [ -f "$z" ] && { unzip -q -o "$z" -d "$(dirname "$out")" && rm -f "$z"; }
    echo "[ok] $f"
  done < "$ROOT/configs/seedvii_kaggle_files.txt"
  [ $fail -eq 0 ] && touch "$DEST/.complete"
  echo "[done] seedvii $(du -sh "$DEST" | cut -f1) fail=$fail"
fi
