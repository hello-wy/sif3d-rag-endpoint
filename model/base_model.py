from utils.registry import Registry
from torch import nn
from model.pointnet_plus2 import PointNet2SemSegSSGShape, MyFPModule
from model.rolling_pose import RollingPose
import torch
from utils.vis_utils import latent_to_joints,visualize_Scene_wo_color, visualize_points, visualize_SMPLXjoints
from model.sif3d_traj import TrajPredictor
from .gcn import GCN
import torch.nn.functional as F
import smplx
from human_body_prior.tools.model_loader import load_vposer

Model = Registry('model')

def create_model(cfg, *args, **kwargs) -> nn.Module:
    """ Create model according to the configuration

    Args:
        cfg: configuration dict
    
    Return:
        Model for prediction
    """ 
    return Model.get(cfg.modelName)(cfg, *args, **kwargs)

class ModelWrapper(nn.Module):
    def __init__(self,config,phase="train"):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vposer, _ = load_vposer(self.config.vposer_path, vp_model='snapshot')
        self.vposer = vposer.to(self.device)
        self.smplx_model = smplx.create(self.config.smplx_path, model_type='smplx', gender='neutral', ext='npz',
                                    num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                    create_betas=True, create_left_hand_pose=True, create_right_hand_pose=True,
                                    create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                    create_reye_pose=True, create_transl=True, num_betas=10, num_expression_coeffs=10,
                                    batch_size=config.batch_size * 16,
                                    ).to(self.device)
        
        self.pointnet = PointNet2SemSegSSGShape({"feat_dim": config.scene_feats_dim})
        self.rollingPose = RollingPose(config)
        
        self.traj_predictor = TrajPredictor(config)
        self.motion_decoder = GCN(config, node_n=69)    # 2M参数量
        
        # 4 metrics
        self.log_vars = nn.Parameter(torch.Tensor([2.0, 2.0, -2.0, -2.0]))
        if phase == "train":
            self.load_ckpts()
        
    def load_ckpts(self):
        state_dict = torch.load(self.config.traj_ckpts)
        print("loaded traj ckpts:" + self.config.traj_ckpts)
        self.traj_predictor.load_state_dict(state_dict, strict=False)
        pointnet_sd = {k[len("pointnet."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("pointnet.")}
        if len(pointnet_sd) > 0:
            self.pointnet.load_state_dict(pointnet_sd, strict=False)

    def forward(self, motions, joints, scene_xyz, gazes, occ):
        scene_feats,_ = self.pointnet(scene_xyz.repeat(1,1,2))
        with torch.no_grad():
            ori, trans = self.traj_predictor(motions, joints, scene_xyz, scene_feats, gazes, occ)     # B,T,3  B,T,3
            trans = torch.cat([trans[:,6:], joints[:,:,0]], dim=1)   # B,T,3
            
        pose = self.rollingPose(motions, joints, scene_xyz, scene_feats.transpose(1, 2).contiguous(), ori, trans)

        recons_joints = latent_to_joints(pose, self.vposer, self.smplx_model)[:, :, :23]
        recons_joints = torch.cat([joints, recons_joints[:, 6:]], dim=1).clone()
        pred_joints = recons_joints[:, :, :23]
            
        pred_joints = self.motion_decoder(pred_joints)
        
        return pose, pred_joints
    
    def _compute_loss(self, poses_label, joints_input, joints_label, motions, joints, scene_xyz, gazes, occ):
        poses_predict, joints_predict = self.forward(motions, joints, scene_xyz, gazes, occ)
        loss_ori, loss_trans, loss_latent, loss_des_ori, loss_des_trans, loss_des_latent, mpjpe, des_mpjpe = \
            self.calculate_loss(poses_predict, poses_label, motions, joints_input[:, :, :23], joints_label[:, :, :23])
        loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = \
            self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])

        # weighted_loss = (torch.exp(-self.log_vars[0]) * loss_trans_gcn.mean() + self.log_vars[0]) + \
        #                 (torch.exp(-self.log_vars[1]) * loss_des_trans_gcn + self.log_vars[1]) + \
        #                 (torch.exp(-self.log_vars[2]) * mpjpe_gcn.mean() + self.log_vars[2]) + \
        #                 (torch.exp(-self.log_vars[3]) * des_mpjpe_gcn + self.log_vars[3])
        
        loss = loss_ori.mean() + loss_trans.mean() + loss_latent.mean() + \
                loss_des_ori + loss_des_trans + loss_des_latent + mpjpe.mean() + des_mpjpe + \
                loss_trans_gcn.mean() + loss_des_trans_gcn + mpjpe_gcn.mean() + des_mpjpe_gcn
        loss = loss + mpjpe_gcn[:, 6:10].mean() + loss_trans_gcn[:, 6:10].mean()
        return loss, mpjpe_gcn, des_mpjpe_gcn, loss_trans_gcn, loss_des_trans_gcn

    
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
        pred_smplx = self.smplx_model(return_verts=True, body_pose=pred_pose,
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
