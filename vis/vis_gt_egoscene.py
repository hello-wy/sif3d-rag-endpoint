
import torch
import time
from torch.utils.data import DataLoader
from dataset import gimo_dataset
from config.config import MotionFromGazeConfig
from model.sif3d_mod import SIF3D
from utils.logger import MetricTracker
from model.partial_scene import HumanGazeSceneUpSample
from utils.vis_utils import visualize_Scene_wo_color,visualize_Scene_wo_color2

"""
PYTHONPATH=./ CUDA_VISIBLE_DEVICES=1 python vis/vis_gt_egoscene.py --sample_points 32768 

"""
class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        self.test_loader = DataLoader(
            gimo_dataset.EgoEvalDataset(config, train=False),
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = HumanGazeSceneUpSample(config)
    
    def vis_ego_scene_by_step(self):
        for i, data in enumerate(self.test_loader):
            gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene = data
            scene_points = scene_points.to(self.device).contiguous()
            joints_input = joints_input.to(self.device)
            joints_label = joints_label.to(self.device)
            if seq[0] == "2022-02-20-044303":
                visualize_Scene_wo_color2(scene_points.cpu().numpy(), "./vis")
                        # 高密度点云、 b，T，3
                # joints = torch.cat([joints_input, joints_label], dim=1)
                # human_points = self.extractor.cut_by_step(scene_points, joints[:, :, 0])      # B,10 // unit,N,3
                # visualize_Scene_wo_color(human_points[0].cpu().numpy(), "./")
            else:
                continue
        
            

    def eval(self):
        model = SIF3D(config, self.vposer, self.body_model).to(self.device)
        model = model.to(self.device)
        assert self.config.load_model_dir is not None
        print('loading pretrained model from ', self.config.load_model_dir)
        state_dict = torch.load(self.config.load_model_dir)
        state_dict = {k: v for k, v in state_dict.items() if "smplx" not in k}
        model.load_state_dict(state_dict, strict=False)
        print('load done!')

        with torch.no_grad():
            model.eval()

            for i, data in enumerate(self.test_loader):
                gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene = data

                # gazes = gazes.to(self.device)
                # poses_input = poses_input.to(self.device)
                # poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)
                if seq[0] == "2022-02-20-044303":
                        # 高密度点云、 b，T，3
                    print(scene_points.shape)
                    joints = torch.cat([joints_input, joints_label], dim=1)
                    human_points = self.extractor.cut_by_step(scene_points, joints[:, :, 0])      # B,10 // unit,N,3
                    
                    # poses_predict, joints_predict = model(poses_input, joints_input[:, :, :23], scene_points, gazes)
                else:
                    continue
               
if __name__ == '__main__':
    config = MotionFromGazeConfig().parse_args()
    start = time.time()
    evaluator = SMPLX_evalutor(config)
    evaluator.vis_ego_scene_by_step()
