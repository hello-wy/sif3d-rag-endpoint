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
from model.sif3d_traj import TrajPredictor
from utils.logger import create_logger, MetricTracker


class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vposer, _ = load_vposer(self.config.vposer_path, vp_model='snapshot')
        self.vposer = vposer.to(self.device)
        self.body_model = smplx.create(self.config.smplx_path, model_type='smplx', gender='neutral', ext='npz',
                                       num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                       create_betas=True, create_left_hand_pose=True, create_right_hand_pose=True,
                                       create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                       create_reye_pose=True, create_transl=True, num_betas=10, num_expression_coeffs=10,
                                       batch_size=config.batch_size * (config.output_seq_len + config.input_seq_len),
                                       ).to(self.device)

        self.model = TrajPredictor(config).to(self.device)

        if self.config.load_model_dir is not None:
            state_dict = torch.load(self.config.load_model_dir)
            state_dict = {k: v for k, v in state_dict.items() if ("smplx_model" not in k and "vposer" not in k)}
            self.model.load_state_dict(state_dict, strict=False)
            print('load done!')

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.lr_adjuster = ExponentialLR(self.optim, gamma=config.gamma)

        self.train_dataset = gimo_dataset.EgoEvalDataset(config, train=True)
        self.test_dataset = gimo_dataset.EgoEvalDataset(config, train=False)
        # exit(0)
        self.best_results = {
            "loss_trans": float("inf"),
            "loss_des_trans": float("inf")
        }
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

        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path, exist_ok=True)
        self.logger = create_logger(config.save_path)
        os.makedirs(f"runs/{self.config.save_path}", exist_ok=True)

    def train(self):
        train_metrics = MetricTracker('loss_trans', 'loss_des_trans', 'loss_vel', 'loss_acc')

        for epoch in range(config.epoch):
            for data in tqdm(self.train_loader):
                gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene, occupancy_map = data

                gazes = gazes.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)
                occupancy_map = occupancy_map.to(self.device)
                
                ori, trans = self.model(poses_input, joints_input[:, :, :23], scene_points, None, gazes, occupancy_map)

                loss_ori, loss_trans, loss_des_ori, loss_des_trans, loss_vel, loss_acc = \
                    self.calculate_loss(ori, trans, poses_label, poses_input, joints_input[:, :, :23], joints_label[:, :, :23])
                # loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = \
                #     self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])

                loss = loss_ori.mean() + loss_trans.mean() + \
                       loss_des_ori + loss_des_trans + \
                       self.config.traj_vel_loss_weight * loss_vel + \
                       self.config.traj_acc_loss_weight * loss_acc
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_metrics.update("loss_trans", loss_trans[:, 6:].mean(), gazes.shape[0])
                train_metrics.update("loss_des_trans", loss_des_trans, gazes.shape[0])
                train_metrics.update("loss_vel", loss_vel, gazes.shape[0])
                train_metrics.update("loss_acc", loss_acc, gazes.shape[0])
                # train_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), gazes.shape[0])
                # train_metrics.update("des_mpjpe", des_mpjpe_gcn, gazes.shape[0])

            self.lr_adjuster.step()
            train_metrics.log(self.logger, epoch)
            train_metrics.reset()

            if epoch % config.val_fre == 0:
                self.model.eval()
                with torch.no_grad():
                    self.test(epoch)
                self.model.train()
            if epoch % config.save_fre == 0:
                torch.save(self.model.state_dict(), f"{self.config.save_path}/{epoch}.pth")

    def test(self, epoch):
        test_metrics = MetricTracker('loss_trans', 'loss_des_trans', 'loss_vel', 'loss_acc')

        for i, data in enumerate(self.test_loader):
            gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene, occupancy_map = data

            gazes = gazes.to(self.device)
            poses_input = poses_input.to(self.device)
            poses_label = poses_label.to(self.device)
            scene_points = scene_points.to(self.device).contiguous()
            joints_input = joints_input.to(self.device)
            joints_label = joints_label.to(self.device)
            occupancy_map = occupancy_map.to(self.device)

            ori, trans = self.model(poses_input, joints_input[:, :, :23], scene_points, None, gazes, occupancy_map)

            loss_ori, loss_trans, loss_des_ori, loss_des_trans, loss_vel, loss_acc = \
                self.calculate_loss(ori,trans,  poses_label, poses_input, joints_input[:, :, :23], joints_label[:, :, :23])

            test_metrics.update("loss_trans", loss_trans[:, 6:].mean(), gazes.shape[0])
            test_metrics.update("loss_des_trans", loss_des_trans, gazes.shape[0])
            test_metrics.update("loss_vel", loss_vel, gazes.shape[0])
            test_metrics.update("loss_acc", loss_acc, gazes.shape[0])


        test_metrics.log(self.logger, epoch)
        current_results = {
            "loss_trans": test_metrics['loss_trans'],
            "loss_des_trans": test_metrics['loss_des_trans']
        }
        test_metrics.reset()
        if all(current_results[key] < self.best_results[key] for key in current_results):
            self.best_results = current_results  # 更新最优值
            self.best_epoch = epoch
            print(f"New best results at epoch {epoch}: {self.best_results}")


    def calculate_loss(self, ori, trans, poses_label, poses_input, joints_input, joints_label):
        poses_label = torch.cat([poses_input, poses_label], dim=1)
        
        # print(poses_label.shape, ori.shape,trans.shape)
        loss_des_ori = F.l1_loss(ori[:, -1, :3], poses_label[:, -1, :3])
        loss_des_trans = torch.norm(trans[:, -1] - poses_label[:, -1, 3:6], dim=-1).mean()

        loss_ori = F.l1_loss(ori, poses_label[:, :, :3], reduction='none')

        loss_trans = torch.norm(trans - poses_label[:, :, 3:6], dim=-1)
        gt_trans = poses_label[:, :, 3:6]
        pred_vel = trans[:, 1:] - trans[:, :-1]
        gt_vel = gt_trans[:, 1:] - gt_trans[:, :-1]
        loss_vel = torch.norm(pred_vel - gt_vel, dim=-1).mean()

        if trans.shape[1] > 2:
            pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
            gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
            loss_acc = torch.norm(pred_acc - gt_acc, dim=-1).mean()
        else:
            loss_acc = trans.new_tensor(0.0)

        return loss_ori, loss_trans, loss_des_ori, loss_des_trans, loss_vel, loss_acc

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
    evaluator.train()
