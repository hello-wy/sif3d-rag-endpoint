import os
import torch
import time
import smplx
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from dataset.gta import DatasetGTA
from human_body_prior.tools.model_loader import load_vposer
from config.config import MotionFromGazeConfig
from model.sif3d_traj_gta import TrajPredictor
from utils.logger import create_logger, MetricTracker


class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = TrajPredictor(config).to(self.device)
        
        state_dict = torch.load("/data/wuyang/output/gta_ckpts/best_traj.pth")
        state_dict = {k: v for k, v in state_dict.items() if ("smplx_model" not in k and "vposer" not in k)}
        self.model.load_state_dict(state_dict, strict=False)
        # if self.config.load_model_dir is not None:
        #     state_dict = torch.load(self.config.load_model_dir)
        #     state_dict = {k: v for k, v in state_dict.items() if ("smplx_model" not in k and "vposer" not in k)}
        #     self.model.load_state_dict(state_dict, strict=False)
        #     print('load done!')

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.lr_adjuster = ExponentialLR(self.optim, gamma=config.gamma)

        self.train_dataset = DatasetGTA(config, train=True)
        self.test_dataset = DatasetGTA(config, train=False)
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
        train_metrics = MetricTracker('loss_trans', 'loss_des_trans')

        for epoch in range(config.epoch):
            for data in tqdm(self.train_loader):
                pose, scene, gazes  = data

                gazes = gazes.to(self.device)
                pose = pose.to(self.device)
                scene = scene.to(self.device).contiguous()
                
                trans = self.model(pose[:,:6], scene, gazes)

                loss_trans, loss_des_trans = \
                    self.calculate_loss(None, trans, pose[:,6:], pose[:,:6], None,None)
                # loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = \
                #     self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])

                loss = loss_trans.mean() + loss_des_trans
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_metrics.update("loss_trans", loss_trans[:, 6:].mean(), gazes.shape[0])
                train_metrics.update("loss_des_trans", loss_des_trans, gazes.shape[0])
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
        test_metrics = MetricTracker('loss_trans', 'loss_des_trans')

        for i, data in enumerate(self.test_loader):
            pose, scene, gazes  = data

            gazes = gazes.to(self.device)
            pose = pose.to(self.device)
            scene = scene.to(self.device).contiguous()
            
            trans = self.model(pose[:,:6], scene, gazes)

            loss_trans, loss_des_trans = \
                self.calculate_loss(None, trans, pose[:,6:], pose[:,:6], None,None)
                
            test_metrics.update("loss_trans", loss_trans[:, 6:].mean(), gazes.shape[0])
            test_metrics.update("loss_des_trans", loss_des_trans, gazes.shape[0])


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
        
        loss_des_trans = torch.norm(trans[:, -1] - poses_label[:, -1, 0], dim=-1).mean()
        loss_trans = torch.norm(trans - poses_label[:, :, 0], dim=-1)

        return loss_trans, loss_des_trans

if __name__ == '__main__':
    config = MotionFromGazeConfig().parse_args()
    start = time.time()

    evaluator = SMPLX_evalutor(config)
    evaluator.train()
