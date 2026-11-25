### useage: stage=1 ckpt=85 bash scripts/eval_traj_robot.sh
### or simply uses best.pth:  bash scripts/eval_traj_robot.sh

#!/usr/bin/env bash

# 读取命令行参数，使用默认值 "best"
ckpt="${1:-${ckpt:-best}}"

# 定义两个阶段的评估脚本命令
# stage1_eval=(
#   "eval_traj_robot.py"                              
#   --batch_size 1                                  
#   --sample_points 4096                            
#   --load_model_dir "/data/wuyang/output/traj_ckpts/${ckpt}.pth" 
#   --modelName TrajPredictor
#   --motion_hidden_dim 128
#   --motion_latent_dim 128
#   --scene_feats_dim 128
#   --gaze_hidden_dim 128
#   --occ                                
# )

stage1_eval=(
  "eval_traj_robot.py"                              
  --batch_size 1                                  
  --sample_points 4096                            
  --load_model_dir "/data/wuyang/output/temp_ckpts/${ckpt}.pth" 
  --modelName TrajPredictor
  --occ                                
)

stage2_eval=(
  "eval_traj_robot.py"  
  --batch_size 1                                  
  --sample_points 4096  
  --load_model_dir "/data/wuyang/output/motion_ckpts/${ckpt}.pth"
  --traj_ckpts "/data/wuyang/output/traj_ckpts/best.pth" 
  --modelName ProgressivePredictor  
)

# 根据环境变量stage选择执行哪一阶段
if [[ "${stage:-1}" == "1" ]]; then
  echo "Running stage 1 with ckpt=${ckpt}"
  CUDA_VISIBLE_DEVICES=1 python "${stage1_eval[@]}"
elif [[ "${stage:-1}" == "2" ]]; then
  echo "Running stage 2 with ckpt=${ckpt}"
  CUDA_VISIBLE_DEVICES=1 python "${stage2_eval[@]}"
else
  echo "Invalid stage. Use stage=1 or stage=2."
  exit 1
fi