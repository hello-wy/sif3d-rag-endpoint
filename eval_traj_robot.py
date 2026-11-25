import torch
import time
import smplx
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import gimo_dataset
from human_body_prior.tools.model_loader import load_vposer
from config.config import MotionFromGazeConfig
from model.base_model import create_model
from utils.logger import MetricTracker
import math

class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.test_dataset = gimo_dataset.EgoEvalDataset(config, train=False)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )

        if self.config.modelName == 'TrajPredictor':
            self.vposer = None
            self.body_model = None
            self.body_model_full = None
            metric_keys = ('loss_trans', 'loss_des_trans')
        else:
            self._init_body_modules()
            metric_keys = ('loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe')

        self.test_metrics = MetricTracker(*metric_keys)
        self.model = self._build_model()
        self._load_checkpoint()

    def _init_body_modules(self):
        self.vposer, _ = load_vposer(self.config.vposer_path, vp_model='snapshot')
        self.vposer = self.vposer.to(self.device)

        common_kwargs = dict(
            model_type='smplx',
            gender='neutral',
            ext='npz',
            num_pca_comps=12,
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_expression=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_transl=True,
            num_betas=10,
            num_expression_coeffs=10,
        )

        self.body_model = smplx.create(
            self.config.smplx_path,
            batch_size=self.config.batch_size * 16,
            **common_kwargs,
        ).to(self.device)
        
        self.body_model_full = smplx.create(
            self.config.smplx_path,
            batch_size=self.config.batch_size * 16,
            **common_kwargs,
        ).to(self.device)

    def _build_model(self):
        if self.config.modelName == 'TrajPredictor':
            return create_model(self.config).to(self.device)
        if self.vposer is None or self.body_model is None:
            raise RuntimeError('vposer and smplx models must be initialised for non-trajectory predictors.')
        return create_model(self.config, self.vposer, self.body_model).to(self.device)

    def _load_checkpoint(self):
        if self.config.load_model_dir is None:
            raise ValueError('load_model_dir must be provided for evaluation.')
        print('loading pretrained model from ', self.config.load_model_dir)
        state_dict = torch.load(self.config.load_model_dir, map_location=self.device)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        skip_tokens = ('smplx', 'vposer')
        filtered_state_dict = {k: v for k, v in state_dict.items() if not any(token in k for token in skip_tokens)}
        self.model.load_state_dict(filtered_state_dict, strict=False)
        print('load done!')

    def eval(self):
        self.model.eval()
        self.test_metrics.reset()

        with torch.no_grad():
            count=0
            for data in self.test_loader:
                gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene, occ = data

                gazes = gazes.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)
                occ = occ.to(self.device)

                if self.config.modelName == 'TrajPredictor':
                    
                    ori, trans = self.model(poses_input, joints_input[:, :, :23], scene_points, gazes, occ)
                    _, loss_trans, _, loss_des_trans = self.calculate_loss(
                        ori,
                        trans,
                        poses_label,
                        poses_input,
                        joints_input[:, :, :23],
                        joints_label[:, :, :23],
                    )

                    if math.dist(trans[0,-1], joints_label[0,-1,0,:3].cpu().numpy()) < 0.5:
                        count +=1
                    self.test_metrics.update('loss_trans', loss_trans[:, 6:].mean(), gazes.shape[0])
                    self.test_metrics.update('loss_des_trans', loss_des_trans, gazes.shape[0])
                else:
                    count=0
                    poses_predict, joints_predict = self.model(
                        poses_input,
                        joints_input[:, :, :23],
                        scene_points,
                        gazes,
                        occ,
                    )
                    loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = self.calc_loss_gcn(
                        joints_predict,
                        joints_label[:, :, :23],
                        joints_input[:, :, :23],
                    )
                    if math.dist(joints_predict[0,-1,0] , joints_label[0,-1,0,:3].cpu().numpy()) < 0.5:
                        count +=1
                    self.test_metrics.update('loss_trans', loss_trans_gcn[:, 6:].mean(), gazes.shape[0])
                    self.test_metrics.update('loss_des_trans', loss_des_trans_gcn, gazes.shape[0])
                    self.test_metrics.update('mpjpe', mpjpe_gcn[:, 6:].mean(), gazes.shape[0])
                    self.test_metrics.update('des_mpjpe', des_mpjpe_gcn, gazes.shape[0])

        results = self.test_metrics.result()
        print(results)
        print('成功率：', count / len(self.test_loader))
        self.test_metrics.reset()
        return results

    def calculate_loss(self, ori, trans, poses_label, poses_input, joints_input, joints_label):
        poses_label = torch.cat([poses_input, poses_label], dim=1)

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
