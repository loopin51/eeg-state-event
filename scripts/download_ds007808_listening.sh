#!/usr/bin/env bash
# ds007808 (OpenNeuro, CC0): listening task only, sub-01 and sub-02 EEG + sidecars, no audio wav (D-057, about 41 GB).
# Run on loopin-ubuntu-server from the project root. Destination is the HDD (D-044); data/raw/ds007808 links to it.
set -euo pipefail
DEST=/mnt/data/eeg_datasets/ds007808
mkdir -p "$DEST"
env/bin/aws s3 sync --no-sign-request s3://openneuro.org/ds007808 "$DEST" \
  --exclude "*" \
  --include "dataset_description.json" --include "README*" --include "participants.*" --include "task-listening_*" \
  --include "sub-01/ses-*/eeg/*task-listening_*" --include "sub-02/ses-*/eeg/*task-listening_*" \
  --include "sub-01/ses-*/eeg/*_electrodes.tsv" --include "sub-01/ses-*/eeg/*_coordsystem.json" \
  --include "sub-02/ses-*/eeg/*_electrodes.tsv" --include "sub-02/ses-*/eeg/*_coordsystem.json"
ln -sfn "$DEST" data/raw/ds007808
find "$DEST" -name "*task-listening_*_eeg.edf" | wc -l
du -sh "$DEST"
