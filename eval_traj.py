import torch
import time
import smplx
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import gimo_dataset
from human_body_prior.tools.model_loader import load_vposer
from config.config import MotionFromGazeConfig
from model.sif3d_traj import SIF3D
from utils.logger import MetricTracker


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
        self.vposer, _ = load_vposer(self.config.vposer_path, vp_model='snapshot')
        self.vposer = self.vposer.to(self.device)

        self.body_model = smplx.create(self.config.smplx_path, model_type='smplx', gender='neutral', ext='npz',
                                       num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                       create_betas=True, create_left_hand_pose=True, create_right_hand_pose=True,
                                       create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                       create_reye_pose=True, create_transl=True, num_betas=10, num_expression_coeffs=10,
                                       batch_size=config.batch_size * 8,
                                       ).to(self.device)
        
        self.body_model_full = smplx.create(self.config.smplx_path, model_type='smplx', gender='neutral', ext='npz',
                                num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                create_betas=True, create_left_hand_pose=True, create_right_hand_pose=True,
                                create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                create_reye_pose=True, create_transl=True, num_betas=10, num_expression_coeffs=10,
                                batch_size=config.batch_size * 16,
                                ).to(self.device)

    def eval(self):
        model = SIF3D(config).to(self.device)
        model = model.to(self.device)
        assert self.config.load_model_dir is not None
        print('loading pretrained model from ', self.config.load_model_dir)
        state_dict = torch.load(self.config.load_model_dir)
        state_dict = {k: v for k, v in state_dict.items() if "smplx" not in k}
        model.load_state_dict(state_dict, strict=False)
        print('load done!')

        with torch.no_grad():
            model.eval()
            test_metrics = MetricTracker('loss_trans', 'loss_des_trans')

            for i, data in enumerate(self.test_loader):
                gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene,occ = data

                gazes = gazes.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)
                occ = occ.to(self.device)

                ori, trans = model(poses_input, joints_input[:, :, :23], scene_points, gazes, occ)


                loss_ori, loss_trans, loss_des_ori, loss_des_trans = \
                    self.calculate_loss(ori,trans,  poses_label, poses_input, joints_input[:, :, :23], joints_label[:, :, :23])

                test_metrics.update("loss_trans", loss_trans[:, 6:].mean(), gazes.shape[0])
                test_metrics.update("loss_des_trans", loss_des_trans, gazes.shape[0])
            print(test_metrics.result())
            test_metrics.reset()

    def calculate_loss(self, ori, trans, poses_label, poses_input, joints_input, joints_label):
        poses_label = torch.cat([poses_input, poses_label], dim=1)
        
        # print(poses_label.shape, ori.shape,trans.shape)
        loss_des_ori = F.l1_loss(ori[:, -1, :3], poses_label[:, -1, :3])
        loss_des_trans = torch.norm(trans[:, -1] - poses_label[:, -1, 3:6], dim=-1).mean()

        loss_ori = F.l1_loss(ori, poses_label[:, :, :3], reduction='none')

        loss_trans = torch.norm(trans - poses_label[:, :, 3:6], dim=-1)

        return loss_ori, loss_trans, loss_des_ori, loss_des_trans
    
    def calc_loss_gcn(self, poses_predict, poses_label, poses_input):
        poses_label = torch.cat([poses_input, poses_label], dim=1)

        loss_trans = torch.norm(poses_predict[:, :, 0] - poses_label[:, :, 0], dim=-1)

        poses_label = poses_label - poses_label[:, :, [0]]
        poses_predict = poses_predict - poses_predict[:, :, [0]]
        mpjpe = torch.norm(poses_predict - poses_label, dim=-1)             # bs * seq * J

        return loss_trans, loss_trans[:, -1].mean(), mpjpe, mpjpe[:, -1].mean()


if __name__ == '__main__':
    config = MotionFromGazeConfig().parse_args()
    start = time.time()
    evaluator = SMPLX_evalutor(config)
    evaluator.eval()
