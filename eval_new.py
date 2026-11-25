import torch
import time
import smplx
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import gimo_dataset
from human_body_prior.tools.model_loader import load_vposer
from config.config import MotionFromGazeConfig
from model.base_model import ModelWrapper
from utils.logger import MetricTracker
from utils.vis_utils import visualize_SMPLXjoints,visualize_gazes
import math
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir="runs/profiler_example")  # You can change the log directory

class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        config.batch_size = 1  # eval batch size is 1
        self.test_loader = DataLoader(
            gimo_dataset.EgoEvalDataset(config, train=False),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    def eval(self):
        model = ModelWrapper(config, "test").to(self.device)
        model = model.to(self.device)
        assert self.config.load_model_dir is not None
        print('loading pretrained model from ', self.config.load_model_dir)
        state_dict = torch.load(self.config.load_model_dir)
        state_dict = {k: v for k, v in state_dict.items() if "smplx" not in k}
        model.load_state_dict(state_dict, strict=False)
        print('load done!')

        with torch.no_grad():
            model.eval()
            test_metrics = MetricTracker('loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe')

            # count = 0
            for i, data in enumerate(self.test_loader):
                gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene, occ = data

                gazes = gazes.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)
                occ = occ.to(self.device)
                
                start_time = time.time()
                _, mpjpe_gcn, des_mpjpe_gcn, loss_trans_gcn, loss_des_trans_gcn = model._compute_loss(
                    poses_label, 
                    joints_input, 
                    joints_label,
                    poses_input, 
                    joints_input[:, :, :23], 
                    scene_points, 
                    gazes, 
                    occ)                
                end_time = time.time()
                if self.config.is_vis:
                    visualize_SMPLXjoints(joints_predict[0,:1].cpu().numpy(),None,i, f'/data/wuyang/output/vis/{scene[0]}/pred_{seq[0]}')
                    visualize_SMPLXjoints(joints_label[0,:1,:23].cpu().numpy(),None,i, f'/data/wuyang/output/vis/{scene[0]}/gt_{seq[0]}')
                    # visualize_SMPLXjoints(joints_predict[0].cpu().numpy(),None,i, f'/data/wuyang/output/vis/{scene[0]}/pred_{seq[0]}')
                    # visualize_SMPLXjoints(joints_label[0,:,:23].cpu().numpy(),None,i, f'/data/wuyang/output/vis/{scene[0]}/gt_{seq[0]}')

                # if math.dist(joints_predict[0,-1,0] , joints_label[0,-1,0,:3].cpu().numpy()) < 0.5:
                #     count +=1
                # loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = \
                #     self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])

                test_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), gazes.shape[0])
                test_metrics.update("loss_des_trans", loss_des_trans_gcn, gazes.shape[0])
                test_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), gazes.shape[0])
                test_metrics.update("des_mpjpe", des_mpjpe_gcn, gazes.shape[0])


                # with open('./pred_seq.csv', 'a', newline='', encoding='utf-8') as f:
                #     writer = csv.writer(f)
                #     writer.writerow([
                #         seq[0],
                #         scene[0],
                #         loss_trans_gcn[:, 6:].cpu().numpy().mean(),
                #         loss_des_trans_gcn.cpu().numpy(),
                #         mpjpe_gcn[:, 6:].cpu().numpy().mean(),
                #         des_mpjpe_gcn.cpu().numpy()
                #     ])
            
            speed = 1.0 / (end_time - start_time)
            print(test_metrics.result())
            test_metrics.reset()
            
            print(f"Speed: {speed:.2f} sample/s")
            # print("成功率：",count/len(self.test_loader))

    def calculate_loss(self, poses_predict, poses_label, poses_input, joints_input, joints_label, test=False):
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
        
        if self.config.eval_len == 1:
            poses_predict = poses_predict[:,:7]
            poses_label = poses_label[:,:7]
        elif self.config.eval_len == 4:
            poses_predict = poses_predict[:,:10]
            poses_label = poses_label[:,:10]
            
        loss_trans = torch.norm(poses_predict[:, :, 0] - poses_label[:, :, 0], dim=-1)

        poses_label = poses_label - poses_label[:, :, [0]]
        poses_predict = poses_predict - poses_predict[:, :, [0]]
        mpjpe = torch.norm(poses_predict - poses_label, dim=-1)             # bs * seq * J

        return loss_trans, loss_trans[:, -1].mean(), mpjpe, mpjpe[:, -1].mean()


if __name__ == '__main__':
    config = MotionFromGazeConfig().parse_args()
    evaluator = SMPLX_evalutor(config)
    evaluator.eval()
