### useage: stage=2 bash scripts/train_traj_robot.sh
### or simply train in stage 1: bash scripts/train_traj_robot.sh


# traj evaluation script for robot motion prediction
# 128 train
# stage1_train=(
#   "train_traj_robot.py"                              
#   --batch_size 8          
#   --sample_points 4096 
#   --modelName TrajPredictor  
#   --dropout 0.4 
#   --epoch 100
#   --save_path /data/wuyang/output/traj_ckpts/ 
#   --occ   
#   --motion_n_heads 4 
#   --motion_hidden_dim 128 
#   --motion_n_layers 1 
#   --motion_latent_dim 128 
#   --scene_feats_dim 128 
#   --gaze_hidden_dim 128 
#   --gaze_n_layers 1         
# )

stage1_train=(
  "train_traj_robot.py"                              
  --batch_size 8          
  --sample_points 4096 
  --modelName TrajPredictor  
  --dropout 0.4 
  --epoch 100
  --save_path /data/wuyang/output/temp_ckpts/    
  --occ                         
)

# motion train
stage2_train=(
    "train_traj_robot.py"
    --batch_size 8
    --sample_points 4096
    --epoch 100
    --dropout 0.4
    --save_path /data/wuyang/output/motion_ckpts/
    --modelName ProgressivePredictor
    --traj_ckpts /data/wuyang/output/traj_ckpts/best.pth   
)

# old_train
# stage2_train=(
#     "train_traj_robot.py"
#     --batch_size 8
#     --sample_points 4096
#     --epoch 150
#     --dropout 0.3
#     --save_path /data/wuyang/output/motion_ckpts_old/
#     --modelName SIF3D
#     --traj_ckpts /data/wuyang/output/traj_ckpts/best.pth   
# )



stage="${1:-${stage:-1}}"
echo "Starting stage ${stage} training..."
if [ "$stage" -eq 1 ]; then
  CUDA_VISIBLE_DEVICES=3 python "${stage1_train[@]}"
  exit 0
fi

CUDA_VISIBLE_DEVICES=3 python "${stage2_train[@]}"