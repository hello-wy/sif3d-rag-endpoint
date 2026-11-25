#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
python scripts/train_rag_refiner.py --rag_refiner_type blend --lr 3e-3 --batch_size 8 \
  --occ --sample_points 32768 "$@"
