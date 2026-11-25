#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
python eval.py \
  --use_rag --rag_train_stage refiner \
  --rag_bank /data/wuyang/output/sif3d_rag_endpoint_20261011/wide_full_terminal/trained.bank.pt \
  --rag_refiner_type adaptive_terminal --rag_refiner_hidden_dim 0 \
  --rag_terminal_max_trust 1.2 --rag_allow_same_sequence --rag_prefer_same_scene \
  --rag_top_k 3 --rag_endpoint_frames 4 \
  --load_model_dir /data/wuyang/output/sif3d_rag_endpoint_20261011/adaptive_terminal_trust12/best.pth \
  --dataroot /data/wuyang/output/sif3d_rag_optimized_20261010/data \
  --dataset_csv dataset --folder cache --sample_points 32768 --occ --batch_size 1 --num_workers 0 \
  --smplx_path /home/customer/sif3d-raw/data --vposer_path /home/customer/sif3d-raw/vposer_v1_0 \
  --output_path /data/wuyang/output/sif3d_rag_endpoint_20261011/reproduced_terminal_171 "$@"
