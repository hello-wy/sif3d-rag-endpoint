from torch.utils.data import Dataset
import numpy as np
import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils.vis_utils import visualize_Mogazejoints
from config.config import MotionFromGazeConfig


class mogaze_dataset(Dataset):
    def __init__(self, config,train = False):
        self.dataroot = config.mogaze_data_dir
        self.config = config
        self.train = train
        split = "train" if train else "test"
        
        self.gazes = torch.load(os.path.join(self.dataroot, split, "gaze_full.pth"),weights_only=True)
        self.poses = torch.load(os.path.join(self.dataroot, split, "pose_full.pth"),weights_only=True)
        self.scene = torch.load(os.path.join(self.dataroot, "scene_1.pth"),weights_only=True)
   
    def __len__(self):
        return len(self.gazes)

    def __getitem__(self, index):
        gazes = self.gazes[index]
        poses = self.poses[index]
        # scene_points = self.scene.clone()
        gazes = gazes[:6].unsqueeze(1)
        
        joints_input = poses[ :self.config.input_seq_len, :, :]
        joints_label = poses[self.config.input_seq_len: , :,  :]
        
        return gazes,joints_input, joints_label, self.scene

        
if __name__ == "__main__":
    config = MotionFromGazeConfig().parse_args()

    train_dataset = mogaze_dataset(config, train=True)
    
    for i in range(113,115):
        set_0,_,a,_ = train_dataset[i]
        print(a.shape)
        
    print("train dataset length: ", len(train_dataset))