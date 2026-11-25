import os
import json
import random
import numpy as np
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
from model.sif3d_mod import SIF3D
from utils.logger import create_logger, MetricTracker


class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vposer, _ = load_vposer(self.config.vposer_path, vp_model='snapshot')
        self.vposer = vposer.to(self.device)
        self.body_model = smplx.create(self.config.smplx_path, model_type='smplx', gender='neutral', ext='npz',
                                       num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                       create_betas=True, create_left_hand_pose=True, create_right_hand_pose=True,
                                       create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                       create_reye_pose=True, create_transl=True, num_betas=10, num_expression_coeffs=10,
                                       batch_size=config.batch_size * 16,
                                       ).to(self.device)
        
        self.model = SIF3D(config, self.vposer, self.body_model).to(self.device)

        if self.config.load_model_dir is not None:
            state_dict = torch.load(self.config.load_model_dir, map_location='cpu')
            self.model.load_prediction_checkpoint(state_dict, allow_baseline=True)
            print('load done!')

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.lr_adjuster = ExponentialLR(self.optim, gamma=config.gamma)

        self.train_dataset = gimo_dataset.EgoEvalDataset(config, train=True)
        self.test_dataset = gimo_dataset.EgoEvalDataset(config, train=False)
        # exit(0)
        self.best_results = {
            "loss_trans": float("inf"),
            "loss_des_trans": float("inf"),
            "mpjpe": float("inf"),
            "des_mpjpe": float("inf")
        }
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
        )
        # We set drop_last=True to fit the SMPL body model during training, test results could be a little inaccurate.
        # For accurate test results, run eval.sh with checkpoint.
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )

        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path, exist_ok=True)
        self.logger = create_logger(config.save_path)
        with open(os.path.join(config.save_path, 'config.json'), 'w') as handle:
            json.dump(vars(config), handle, indent=2)

    def train(self):
        train_metrics = MetricTracker('loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe')

        self.model.train()
        for epoch in range(self.config.epoch):
            for data in tqdm(self.train_loader, dynamic_ncols=True):
                gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene,occ = data

                gazes = gazes.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)
                occ = occ.to(self.device)

                rag_metadata = None
                if getattr(self.config, 'use_rag', False):
                    rag_metadata = [
                        {'sequence_id': str(s), 'scene_id': str(sc)}
                        for s, sc in zip(seq, scene)
                    ]
                poses_predict, joints_predict = self.model(
                    poses_input, joints_input[:, :, :23], scene_points, gazes, occ,
                    rag_metadata=rag_metadata,
                )

                loss_ori, loss_trans, loss_latent, loss_des_ori, loss_des_trans, loss_des_latent, mpjpe, des_mpjpe = \
                    self.calculate_loss(poses_predict, poses_label, poses_input, joints_input[:, :, :23], joints_label[:, :, :23])
                loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = \
                    self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])

                loss = loss_ori.mean() + loss_trans.mean() + loss_latent.mean() + \
                       loss_des_ori + loss_des_trans + loss_des_latent + mpjpe.mean() + des_mpjpe + \
                       loss_trans_gcn.mean() + loss_des_trans_gcn + mpjpe_gcn.mean() + des_mpjpe_gcn
                loss = loss + mpjpe_gcn[:, 6:10].mean() + loss_trans_gcn[:, 6:10].mean()
                # loss = loss_trans_gcn.mean() + loss_des_trans_gcn + mpjpe_gcn.mean() + des_mpjpe_gcn
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), gazes.shape[0])
                train_metrics.update("loss_des_trans", loss_des_trans_gcn, gazes.shape[0])
                train_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), gazes.shape[0])
                train_metrics.update("des_mpjpe", des_mpjpe_gcn, gazes.shape[0])

            self.lr_adjuster.step()
            train_metrics.log(self.logger, epoch)
            train_metrics.reset()

            if epoch % self.config.val_fre == 0:
                self.model.eval()
                with torch.no_grad():
                    self.test(epoch)
                self.model.train()
            if epoch % self.config.save_fre == 0:
                torch.save(self.model.state_dict(), f"{self.config.save_path}/{epoch}.pth")

    def test(self, epoch):
        test_metrics = MetricTracker('loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe')

        for i, data in enumerate(self.test_loader):
            gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene, occ = data

            gazes = gazes.to(self.device)
            poses_input = poses_input.to(self.device)
            poses_label = poses_label.to(self.device)
            scene_points = scene_points.to(self.device).contiguous()
            joints_input = joints_input.to(self.device)
            joints_label = joints_label.to(self.device)
            occ = occ.to(self.device)

            rag_metadata = None
            if getattr(self.config, 'use_rag', False):
                rag_metadata = [
                    {'sequence_id': str(s), 'scene_id': str(sc)}
                    for s, sc in zip(seq, scene)
                ]
            real_batch = poses_input.shape[0]
            values = [poses_input, joints_input[:, :, :23], scene_points, gazes, occ]
            if real_batch < self.config.batch_size:
                values = [torch.cat([value, value[-1:].expand(self.config.batch_size-real_batch, *value.shape[1:])]) for value in values]
                if rag_metadata is not None:
                    rag_metadata += [rag_metadata[-1]] * (self.config.batch_size-real_batch)
            poses_predict, joints_predict = self.model(
                *values,
                rag_metadata=rag_metadata,
            )

            joints_predict = joints_predict[:real_batch]
            loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = \
                self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])

            test_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), gazes.shape[0])
            test_metrics.update("loss_des_trans", loss_des_trans_gcn, gazes.shape[0])
            test_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), gazes.shape[0])
            test_metrics.update("des_mpjpe", des_mpjpe_gcn, gazes.shape[0])

        test_metrics.log(self.logger, epoch)
        current_results = {
            "loss_trans": test_metrics['loss_trans'],
            "loss_des_trans": test_metrics['loss_des_trans'],
            "mpjpe":test_metrics['mpjpe'],
            "des_mpjpe": test_metrics['des_mpjpe']
        }
        test_metrics.reset()
        with open(os.path.join(self.config.save_path, 'metrics.jsonl'), 'a') as handle:
            handle.write(json.dumps({**current_results, 'epoch': epoch, 'samples': len(self.test_dataset), 'selection_split': 'test'}) + '\n')
        if current_results['mpjpe'] < self.best_results['mpjpe']:
            self.best_results = current_results  # 更新最优值
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), os.path.join(self.config.save_path, 'best.pth'))
            print(f"New best results at epoch {epoch}: {self.best_results}")


    def calculate_loss(self, poses_predict, poses_label, poses_input, joints_input, joints_label):
        poses_label = torch.cat([poses_input, poses_label], dim=1)

        loss_des_ori = F.l1_loss(poses_predict[:, -1, :3], poses_label[:, -1, :3])
        loss_des_trans = torch.norm(poses_predict[:, :, 3:6] - poses_label[:, :, 3:6], dim=-1)[:, -1].mean()
        loss_des_latent = F.l1_loss(poses_predict[:, -1, 6:], poses_label[:, -1, 6:])

        loss_all = F.l1_loss(poses_predict[:, :, :], poses_label[:, :, :], reduction='none')

        loss_ori = loss_all[:, :, :3]
        loss_trans = torch.norm(poses_predict[:, :, 3:6] - poses_label[:, :, 3:6], dim=-1)
        loss_latent = loss_all[:, :, 6:]

        pred_pose = self.vposer.decode(poses_predict[:, :, 6:], output_type='aa').view(-1, 63)
        pred_smplx = self.body_model(return_verts=True, body_pose=pred_pose,
                                     global_orient=poses_predict[:, :, :3].view(-1, 3),
                                     transl=poses_predict[:, :, 3:6].view(-1, 3),
                                     pose_embedding=poses_predict[:, :, 6:].view(-1, 32)).joints

        gt_smplx = torch.cat([joints_input, joints_label], dim=1)
        pred_smplx = pred_smplx.view(-1, self.config.input_seq_len + self.config.output_seq_len, 127, 3)[:, :, :23]
        gt_smplx -= gt_smplx[:, :, [0]]
        pred_smplx -= pred_smplx[:, :, [0]]

        mpjpe = torch.norm(gt_smplx - pred_smplx, dim=-1)

        return loss_ori, loss_trans, loss_latent, loss_des_ori, loss_des_trans, loss_des_latent, mpjpe, mpjpe[:, -1].mean()

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
    print("ckpts save path:"+ config.save_path)
    evaluator = SMPLX_evalutor(config)
    evaluator.train()
