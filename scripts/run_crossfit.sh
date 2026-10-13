#!/usr/bin/env bash
# D-027 cross-fitting: train Stage A once per outer subject fold (sequentially; one 12GB GPU fits one run),
# encode every window with each fold encoder, then evaluate each subject only through the encoder that never saw it.
# Usage (on the training host): bash scripts/run_crossfit.sh configs/<base>.yaml [N_FOLDS=5]
set -u
cd "$(dirname "$0")/.."
BASE="$1"; K="${2:-5}"; N="$(basename "$BASE" .yaml)"; STAMP="$(date +%Y%m%d_%H%M)"; RUNS=()
for F in $(seq 0 $((K-1))); do
  mkdir -p results/crossfit_configs; CFG="results/crossfit_configs/${N}_f${F}.yaml"   # not configs/: remote_setup.sh code runs rsync --delete
  ./env/bin/python - "$BASE" "$CFG" "$K" "$F" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); c['n_folds'] = int(sys.argv[3]); c['fold'] = int(sys.argv[4]); c['name'] = c['name'] + f"_cf{sys.argv[4]}"
yaml.safe_dump(c, open(sys.argv[2], 'w'), sort_keys=False)
PY
  R="results/runs/${STAMP}_${N}_cf${F}"; mkdir -p "$R"; RUNS+=("$R")
  echo "START fold $F $R"
  ./env/bin/python scripts/train_stageA.py "$CFG" --out "$R" > "$R/train.log" 2>&1 || { echo "FAIL train $R"; continue; }
  ./env/bin/python scripts/eval_stageB.py "$R" --extract-only > "$R/extract.log" 2>&1 || echo "FAIL extract $R"
  echo "DONE fold $F $R"
done
./env/bin/python scripts/eval_crossfit.py "results/crossfit_${STAMP}_${N}.json" "${RUNS[@]}" > "results/crossfit_${STAMP}_${N}.log" 2>&1
echo "CROSSFIT_DONE results/crossfit_${STAMP}_${N}.log"
