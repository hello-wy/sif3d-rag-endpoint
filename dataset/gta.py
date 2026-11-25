import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from dataclasses import dataclass
import tyro

@dataclass
class gtaConfig:
    dataroot: str = "/data/wuyang/GTA-IM/data_v2_downsample0.02"
    input_seq_len: int = 6
    output_seq_len: int = 10
    num_scene_points: int = 5000
    max_dist_from_human: float = 2.0  # meter


class DatasetGTA(Dataset):

    def __init__(self, config, train=False):
        self.mode = "train" if train else "test"
        self.config = config
        self.train = train
        self.dataroot = "/data/wuyang/GTA-IM"
        self.input_seq_len = config.input_seq_len
        self.output_seq_len = config.output_seq_len

        self.data = torch.load(os.path.join(self.dataroot, f'processed/{self.mode}/gta_processed_{self.mode}.pt'))

    def __len__(self):
        return len(self.data['pose'])


    def __getitem__(self, idx):
        pose = torch.from_numpy(self.data['pose'][idx]).float()
        scene = self.data['scene'][idx]
        gaze = torch.from_numpy(self.data['gaze_list'][idx]).float()
    
        return pose, scene, gaze

if __name__ == '__main__':
    config = tyro.cli(gtaConfig)
    
    # 测试训练集
    print("\n=== 加载训练集 ===")
    train_dataset = DatasetGTA(config, train=True)
    print(f"训练集大小: {len(train_dataset)}")
    

    pose, scene, gaze = train_dataset[0]
    pose, scene, gaze = train_dataset[1]

    
    # 使用DataLoader
    # from torch.utils.data import DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    

