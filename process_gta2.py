import tyro
from dataclasses import dataclass
import numpy as np
import os
import torch
from tqdm import tqdm
from pointnet2_ops import pointnet2_utils

@dataclass
class gtaConfig:
    mode: str = "train"
    input_seq_len: int = 6
    output_seq_len: int = 10
    num_scene_points: int = 8192


def farthest_point_sampling(points, num_samples):
    """
    最远点采样算法
    Args:
        points: (N, 3) 点云数据
        num_samples: 采样点数
    Returns:
        sampled_points: (num_samples, 3) 采样后的点云
    """
    N = points.shape[0]
    if N <= num_samples:
        # 如果点数不足，直接返回或填充
        if N < num_samples:
            # 重复采样填充
            indices = np.random.choice(N, num_samples, replace=True)
            return points[indices]
        return points
    
    centroids = np.zeros(num_samples, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    
    return points[centroids]


def compute_gaze_from_joints(joints, num_frames=6):
    """
    根据joints计算gaze点
    Args:
        joints: (T, J, 3) joints数据，T为帧数，J为关节数
        num_frames: 需要计算的帧数
    Returns:
        gaze_points: (num_frames, 1, 3) gaze点
    """
    # 假设使用头部关节(通常是第15个关节)来计算gaze方向
    # 这里使用头部位置和颈部位置来计算视线方向
    head_idx = 15  # SMPL头部关节索引
    neck_idx = 12  # SMPL颈部关节索引
    
    gaze_points = []
    for i in range(min(num_frames, joints.shape[0])):
        head_pos = joints[i, head_idx]
        neck_pos = joints[i, neck_idx]
        
        # 计算头部朝向
        direction = head_pos - neck_pos
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        # gaze点设置为头部位置加上朝向的延伸
        gaze_point = head_pos + direction * 1.0  # 延伸1米
        gaze_points.append(gaze_point)
    
    gaze_points = np.array(gaze_points)  # (num_frames, 3)
    gaze_points = gaze_points[:, np.newaxis, :]  # (num_frames, 1, 3)
    
    return gaze_points


# 处理gaze点以及scene 点云数量统一、pose 降采样
def process_gta(pose_data, scene_data, config, target_frames=16):
    """
    处理GTA数据
    Args:
        pose_data: (T, J, 3) 原始pose数据
        scene_data: (N, 3) 原始场景点云数据
        config: 配置对象
        target_frames: 目标帧数
    Returns:
        processed_pose: (target_frames, J, 3) 降采样后的pose
        processed_scene: (num_scene_points, 3) 采样后的场景点云
        gaze_points: (6, 1, 3) 前6帧的gaze点
    """
    # 1. pose降采样到16帧
    original_frames = pose_data.shape[0]
    if original_frames >= target_frames:
        # 均匀采样16帧
        indices = np.linspace(0, original_frames - 1, target_frames, dtype=np.int32)
        processed_pose = pose_data[indices]
    else:
        # 如果原始帧数不足，进行插值或重复
        indices = np.linspace(0, original_frames - 1, target_frames)
        processed_pose = np.array([pose_data[int(idx)] for idx in indices])
    
    # 2. 场景点云最远点采样
    scene_data = torch.from_numpy(scene_data).float().cuda().unsqueeze(0)
    new_idx = pointnet2_utils.furthest_point_sample(scene_data, config.num_scene_points).squeeze().long().cpu()
    processed_scene = scene_data.squeeze().cpu()[new_idx]    
    
    # 3. 根据前6帧计算gaze点
    gaze_points = compute_gaze_from_joints(processed_pose, num_frames=config.input_seq_len)
    
    return processed_pose, processed_scene, gaze_points

def main(config):
    root = "/data/wuyang/GTA-IM/data_v2_downsample0.02"
    scene_split = {'train': ['r001','r002','r003', 'r006'],
                    'test': ['r010', 'r011', 'r013']
                }
    
    pose = {}
    scene = {}
    idx2scene = {}
    gaze_list = {}
    
    # 获取所有需要处理的序列
    all_seqs = sorted(os.listdir(root))
    seqs_to_process = [seq for seq in all_seqs if seq.split('_')[1] in scene_split[config.mode]]
    
    print(f"开始处理 {config.mode} 模式数据，共 {len(seqs_to_process)} 个序列")
    
    idx = 0
    for seq in tqdm(seqs_to_process, desc=f"处理{config.mode}数据"):
        data_tmp = np.load(os.path.join(root, seq), allow_pickle=True)
        
        # 处理数据
        processed_pose, processed_scene, gaze_points = process_gta(
            data_tmp['joints'], 
            data_tmp['scene_points'], 
            config,
            target_frames=16
        )
        
        pose[idx] = processed_pose
        scene[idx] = processed_scene
        gaze_list[idx] = gaze_points
        idx2scene[idx] = seq[:-4]
        
        idx += 1
    
    # 保存处理后的数据
    output_dir = os.path.join(root, f"/data/wuyang/GTA-IM/processed/{config.mode}")
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save({
        'pose': pose,
        'scene': scene,
        'gaze_list': gaze_list,
        'idx2scene': idx2scene
    }, os.path.join(output_dir, f'gta_processed_{config.mode}.pt'))
    
    print(f"Saved processed data to {output_dir}")
    print(f"Total sequences: {idx}")
    print(f"✓ 处理完成！")
    

if __name__ == "__main__":
    main(tyro.cli(gtaConfig))
        

        

    

