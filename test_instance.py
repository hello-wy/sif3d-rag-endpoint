import numpy as np
from utils.vis_utils import visualize_Scene_wo_color
import trimesh
import torch


scenes = torch.load('/data/wuyang/human/GIMO/318/train/scene_points_4096.pth')
bedroom = scenes[1]
# ins = trimesh.load('/data/wuyang/results_cgimo/ply_each_instance/bedroom0122_p1054.ply')
# ins = torch.from_numpy(ins.vertices).float()

visualize_Scene_wo_color(bedroom,'bed')