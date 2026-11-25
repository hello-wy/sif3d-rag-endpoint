CUDA_VISIBLE_DEVICES=1 python train_traj.py --save_fre 1 --val_fre 1 --batch_size 8 --sample_points 32768 \
    --dropout 0.3 --epoch 200 --save_path /data/wuyang/output/traj_ckpts/