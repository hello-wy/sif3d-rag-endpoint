import os
import torch
import time
import smplx
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from dataset import gimo_dataset
from human_body_prior.tools.model_loader import load_vposer
from config.config import MotionFromGazeConfig
from model.base_model import create_model
from utils.logger import create_logger, MetricTracker

class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_dataset = gimo_dataset.EgoEvalDataset(config, train=True)
        self.test_dataset = gimo_dataset.EgoEvalDataset(config, train=False)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )
        # We set drop_last=True to fit the SMPL body model during training, test results could be a little inaccurate.
        # For accurate test results, run eval.sh with checkpoint.
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )

        if self.config.modelName == 'TrajPredictor':
            self.vposer = None
            self.body_model = None
            self.body_model_full = None
            metric_keys = ('loss_trans', 'loss_des_trans')
        else:
            self._init_body_modules()
            metric_keys = ('loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe')
        self.best_results = {key: float('inf') for key in metric_keys}
        self.train_metrics = MetricTracker(*metric_keys)
        self.test_metrics = MetricTracker(*metric_keys)

        self.model = self._build_model()

        if self.config.load_model_dir is not None:
            state_dict = torch.load(self.config.load_model_dir)
            state_dict = {k: v for k, v in state_dict.items() if ("smplx_model" not in k and "vposer" not in k)}
            self.model.load_state_dict(state_dict, strict=False)
            print('load done!')

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.lr_adjuster = ExponentialLR(self.optim, gamma=config.gamma)
        
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path, exist_ok=True)
        self.logger = create_logger(config.save_path)
        self._log_model_parameters()

    def _log_model_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")
        
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
    
    def train(self):
        for epoch in range(self.config.epoch):
            self.train_metrics.reset()
            self.training(self.train_loader, epoch, is_train=True)

            self.lr_adjuster.step()
            self.train_metrics.log(self.logger, epoch, train=True)

            if epoch % self.config.val_fre == 0:
                self.model.eval()
                with torch.no_grad():
                    self.test(epoch)
                self.model.train()
            if epoch % self.config.save_fre == 0:
                torch.save(self.model.state_dict(), f"{self.config.save_path}/{epoch}.pth")

    def test(self, epoch):
        current_results = self.training(self.test_loader, epoch, is_train=False)

        self.test_metrics.log(self.logger, epoch, train=False)

        if all(current_results[key] < self.best_results[key] for key in current_results):
            self.best_results = current_results  # 更新最优值
            self.best_epoch = epoch
            print(f"New best results at epoch {epoch}: {self.best_results}")

        self.test_metrics.reset()
        return current_results

    def training(self, dataloader, epoch, is_train):
        metrics = self.train_metrics if is_train else self.test_metrics

        for data in dataloader:
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

                loss_ori, loss_trans, loss_des_ori, loss_des_trans = self.calculate_loss(
                    ori,
                    trans,
                    poses_label,
                    poses_input,
                    joints_input[:, :, :23],
                    joints_label[:, :, :23],
                )

                if is_train:
                    loss = loss_ori.mean() + loss_trans.mean() + loss_des_ori + loss_des_trans
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                metrics.update('loss_trans', loss_trans[:, 6:].mean(), gazes.shape[0])
                metrics.update('loss_des_trans', loss_des_trans, gazes.shape[0])

            else:
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

                if is_train:
                    loss_ori, loss_trans, loss_latent, loss_des_ori, loss_des_trans, loss_des_latent, mpjpe, des_mpjpe = \
                        self.calculate_motion_loss(poses_predict, poses_label, poses_input, joints_input[:, :, :23], joints_label[:, :, :23])
                    loss = loss_ori.mean() + loss_trans.mean() + loss_latent.mean() + \
                       loss_des_ori + loss_des_trans + loss_des_latent + mpjpe.mean() + des_mpjpe + \
                       loss_trans_gcn.mean() + loss_des_trans_gcn + mpjpe_gcn.mean() + des_mpjpe_gcn
                       
                    # loss = loss_trans_gcn.mean() + loss_des_trans_gcn + mpjpe_gcn.mean() + des_mpjpe_gcn

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                metrics.update('loss_trans', loss_trans_gcn[:, 6:].mean(), gazes.shape[0])
                metrics.update('loss_des_trans', loss_des_trans_gcn, gazes.shape[0])
                metrics.update('mpjpe', mpjpe_gcn[:, 6:].mean(), gazes.shape[0])
                metrics.update('des_mpjpe', des_mpjpe_gcn, gazes.shape[0])

        return metrics.result()
    
    # motion 
    def calculate_motion_loss(self, poses_predict, poses_label, poses_input, joints_input, joints_label):
        poses_label = torch.cat([poses_input, poses_label], dim=1)

        loss_des_ori = F.l1_loss(poses_predict[:, -1, :3], poses_label[:, -1, :3])
        loss_des_trans = torch.norm(poses_predict[:, :, 3:6] - poses_label[:, :, 3:6], dim=-1)[:, -1].mean()
        loss_des_latent = F.l1_loss(poses_predict[:, -1, 6:], poses_label[:, -1, 6:])

        loss_all = F.l1_loss(poses_predict[:, :, :], poses_label[:, :, :], reduction='none')

        loss_ori = loss_all[:, :, :3]
        loss_trans = torch.norm(poses_predict[:, :, 3:6] - poses_label[:, :, 3:6], dim=-1)
        loss_latent = loss_all[:, :, 6:]

        pred_pose = self.vposer.decode(poses_predict[:, :, 6:], output_type='aa').view(-1, 63)
        pred_smplx = self.body_model_full(return_verts=True, body_pose=pred_pose,
                                     global_orient=poses_predict[:, :, :3].view(-1, 3),
                                     transl=poses_predict[:, :, 3:6].view(-1, 3),
                                     pose_embedding=poses_predict[:, :, 6:].view(-1, 32)).joints

        gt_smplx = torch.cat([joints_input, joints_label], dim=1)
        pred_smplx = pred_smplx.view(-1, self.config.input_seq_len + self.config.output_seq_len, 127, 3)[:, :, :23]
        gt_smplx -= gt_smplx[:, :, [0]]
        pred_smplx -= pred_smplx[:, :, [0]]

        mpjpe = torch.norm(gt_smplx - pred_smplx, dim=-1)

        return loss_ori, loss_trans, loss_latent, loss_des_ori, loss_des_trans, loss_des_latent, mpjpe, mpjpe[:, -1].mean()
    
    # traj
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

    evaluator = SMPLX_evalutor(config)
    evaluator.train()
