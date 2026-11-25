import os
import torch
import time
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from dataset import gimo_dataset
from config.config import MotionFromGazeConfig
from model.base_model import ModelWrapper 
from utils.logger import create_logger, MetricTracker
from utils.vis_utils import visualize_SMPLXjoints

class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ModelWrapper(config).to(self.device)

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
            "loss_des_trans": float("inf"),
            "mpjpe": float("inf"),
            "des_mpjpe": float("inf")
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
        self._log_model_parameters()

    def _log_model_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")
        
    def train(self):
        train_metrics = MetricTracker('loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe')

        for epoch in range(config.epoch):
            for data in tqdm(self.train_loader, dynamic_ncols=True):
                gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene,occ = data

                gazes = gazes.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)
                occ = occ.to(self.device)

                # poses_predict, joints_predict = self.model()
                loss, mpjpe_gcn, des_mpjpe_gcn, loss_trans_gcn, loss_des_trans_gcn = self.model._compute_loss(
                    poses_label, 
                    joints_input, 
                    joints_label,
                    poses_input, 
                    joints_input[:, :, :23], 
                    scene_points, 
                    gazes, 
                    occ)
                
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

            if epoch % config.val_fre == 0 and epoch > 20:
                self.model.eval()
                with torch.no_grad():
                    self.test(epoch)
                self.model.train()
            if epoch % config.save_fre == 0 and epoch > 20:
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

            poses_predict, joints_predict = self.model(poses_input, joints_input[:, :, :23], scene_points, gazes, occ)

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
        if all(current_results[key] < self.best_results[key] for key in current_results):
            self.best_results = current_results  # 更新最优值
            self.best_epoch = epoch
            print(f"New best results at epoch {epoch}: {self.best_results}")
            
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
