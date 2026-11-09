#!/bin/bash
# Run full DHCP inference across all output types, models, and pairs
# Usage: bash run_all.sh [GPU_ID]
#   e.g. bash run_all.sh 3

GPU_ID=${1:-0}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="python3"

echo "============================================"
echo "  DHCP Brain Age Inference - Full Run"
echo "  GPU: ${GPU_ID}"
echo "  Start: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} ${SCRIPT_DIR}/run_inference.py 2>&1 | tee ${SCRIPT_DIR}/run_all.log

echo ""
echo "============================================"
echo "  Finished: $(date)"
echo "============================================"
