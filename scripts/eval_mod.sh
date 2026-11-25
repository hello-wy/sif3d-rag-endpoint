
# CUDA_VISIBLE_DEVICES=3 python eval_mod.py --batch_size 1 --sample_points 4096 \
#  --load_model_dir /data/wuyang/output/temp_ckpts/99.pth --occ "$@"

CUDA_VISIBLE_DEVICES=3 python eval_mod.py --batch_size 1 --sample_points 4096 \
 --load_model_dir /data/wuyang/output/motion_ckpts/best.pth --occ "$@"

# CUDA_VISIBLE_DEVICES=3 python eval_mod.py --batch_size 1 --sample_points 4096 \
#  --load_model_dir /data/wuyang/output/traj_ckpts/28.pth --occ "$@"
# CUDA_VISIBLE_DEVICES=3 python eval_mod.py --batch_size 1 --sample_points 4096 \
#  --load_model_dir /data/wuyang/output/ablation_ckpts/53.pth --occ "$@"
