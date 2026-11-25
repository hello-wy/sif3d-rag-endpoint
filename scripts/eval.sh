CUDA_VISIBLE_DEVICES=1 python eval.py --batch_size 1 --sample_points 32768 --batch_size 8 \
    --load_model_dir /data/wuyang/output/motion_ckpts/8.pth
    #  --load_model_dir ./sif3d_checkpoint.pth