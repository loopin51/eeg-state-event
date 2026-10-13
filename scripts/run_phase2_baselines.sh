#!/usr/bin/env bash
# Phase 2 remaining baselines under the cross-fit splits (roadmap 2.8, 2.9). GPU steps run sequentially.
#   1) CBraMod / LaBraM frozen features  2) read-out via eval_crossfit (EXTRA_FEATURES)  3) EEGNet cross-fit
# Usage (training host): bash scripts/run_phase2_baselines.sh <crossfit_stamp_name> [WAIT_PID]
#   e.g. bash scripts/run_phase2_baselines.sh 20260923_1810_clisa_align_only 12345
set -u
cd "$(dirname "$0")/.."
TAG="$1"; WAIT="${2:-}"
if [ -n "$WAIT" ]; then echo "waiting for pid $WAIT"; while kill -0 "$WAIT" 2>/dev/null; do sleep 60; done; fi
RUNS=$(ls -d results/runs/${TAG}_cf* | sort)
for M in cbramod labram; do ./env/bin/python scripts/extract_fm_features.py --model $M || echo "FAIL extract $M"; done
EXTRA_FEATURES=cbramod,labram ./env/bin/python scripts/eval_crossfit.py results/crossfit_fm_${TAG}.json $RUNS > results/crossfit_fm_${TAG}.log 2>&1 || echo "FAIL eval fm"
./env/bin/python scripts/eegnet_crossfit.py results/crossfit_${TAG}.json results/crossfit_eegnet_${TAG}.json $RUNS > results/crossfit_eegnet_${TAG}.log 2>&1 || echo "FAIL eegnet"
echo "PHASE2_BASELINES_DONE"
