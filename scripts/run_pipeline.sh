#!/usr/bin/env bash
# Run train -> Stage B -> factorization diagnosis for one or more configs, SEQUENTIALLY.
# Lives in the repo so `remote_setup.sh code` (rsync --delete) does not remove it.
# Usage (on the training host):  bash scripts/run_pipeline.sh configs/a.yaml [configs/b.yaml ...]
# A single 12GB GPU fits exactly one run at a time (~10GB), so never launch these in parallel.
set -u
cd "$(dirname "$0")/.."
for CFG in "$@"; do
  N="$(basename "$CFG" .yaml)"
  R="results/runs/$(date +%Y%m%d_%H%M)_${N}"
  mkdir -p "$R"
  echo "START $R"
  ./env/bin/python scripts/train_stageA.py "$CFG" --out "$R" > "$R/train.log" 2>&1 || { echo "FAIL train $R"; continue; }
  ./env/bin/python scripts/eval_stageB.py "$R"            > "$R/stageB.log" 2>&1 || echo "FAIL eval $R"
  ./env/bin/python scripts/diagnose_factorization.py "$R" > "$R/factorization.log" 2>&1 || echo "FAIL diag $R"
  echo "DONE $R"
done
