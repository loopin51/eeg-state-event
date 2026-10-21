#!/usr/bin/env bash
# D-113 arms a and b in one gpuq CPU job (order of reports/08_fv_stage/notes/gao_ab_implementation_kr.md section 6).
# usage (gpuq, from the project folder): bash scripts/fv_gao_run.sh configs/fv_gao.yaml
set -euo pipefail
CFG=$1
env/bin/python -I scripts/fv_gao_prep.py "$CFG"
env/bin/python -I scripts/fv_gao_erp.py "$CFG"
env/bin/python -I scripts/fv_gao_decode.py "$CFG" --arm b
env/bin/python -I scripts/fv_gao_decode.py "$CFG" --arm a
