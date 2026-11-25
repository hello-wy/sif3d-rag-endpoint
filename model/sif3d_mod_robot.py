"""Readable re-write of the SIF3D module preserving the original logic."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_cross_model import PerceiveEncoder, SelfAttentionLayer
from model.gcn import GCN
from model.partial_scene import HumanGazeSceneUpSample
from model.pointnet_plus2 import MyFPModule, PointNet2SemSegSSGShape
from utils.vis_utils import latent_to_joints
from model.base_model import Model
import numpy as np
Tensor = torch.Tensor


def human_centered_scene(xyz: Tensor, ori: Tensor, trans: Tensor) -> Tensor:
    """
    Normalize scene points to the local human frame.

    Args:
        xyz: Scene points `[B, N, 3]`.
        ori: Human global orientations `[B, T, 3]`.
        trans: Human translations `[B, T, 3]`.

    Returns:
        Scene points expressed in the human-centric frame `[B, T, N, 3]`.
    """
    batch_size, seq_len, _ = ori.shape
    sin_ori = torch.sin(ori[:, :, 1]).unsqueeze(-1)
    cos_ori = torch.cos(ori[:, :, 1]).unsqueeze(-1)
    centered_xyz = xyz[:, None, :, :].repeat(1, seq_len, 1, 1) - trans[:, :, None, :]

    rotation = torch.cat(
        [
            torch.cat([sin_ori, torch.zeros_like(sin_ori), -cos_ori], dim=-1),
            torch.cat(
                [torch.zeros_like(sin_ori), torch.ones_like(sin_ori), torch.zeros_like(sin_ori)],
                dim=-1,
            ),
            torch.cat([cos_ori, torch.zeros_like(sin_ori), sin_ori], dim=-1),
        ],
        dim=-1,
    )
    rotation = rotation.view(batch_size, seq_len, 3, 3)
    return torch.matmul(centered_xyz, rotation)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.nfeats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(input_feats, latent_dim)

    def forward(self, x):
        x = x.permute((1, 0, 2))
        x = self.poseEmbedding(x)  # --> [seqlen, bs, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.poseFinal = nn.Linear(latent_dim, input_feats)

    def forward(self, output):
        output = self.poseFinal(output)  # [seqlen, bs, nfeats]
        output = output.permute(1, 0, 2)  # [bs, seqlen, nfeats]
        return output


class Mlp(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, expansion: int = 4, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim * expansion)
        self.fc2 = nn.Linear(out_dim * expansion, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.drop(F.gelu(self.fc1(x)))
        return self.drop(self.fc2(hidden))


class TIABlock(nn.Module):
    def __init__(self, feat_dim: int, num_head: int = 4) -> None:
        super().__init__()
        self.motion_encoder = SelfAttentionLayer(num_heads=8, num_q_channels=feat_dim, dropout=0.0)
        self.ia_attention = IntentionAwareAttention(feat_dim, num_head)
        self.mlp = Mlp(feat_dim * 3, feat_dim, drop=0.0)

    def forward(
        self,
        motion: Tensor,
        scene_xyz: Tensor,
        scene_feats: Tensor,
        gaze_embedding: Tensor,
    ) -> Tensor:
        motion_tia = self.motion_encoder(motion)
        scene_motion_feats = self.ia_attention(motion_tia, scene_feats)
        fused = torch.cat([motion_tia, scene_motion_feats, gaze_embedding], dim=-1)
        return self.mlp(fused)


class IntentionAwareAttention(nn.Module):
    def __init__(self, feat_dim: int, num_head: int = 4) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.motion_q = nn.Linear(feat_dim, feat_dim)
        self.scene_kv = nn.Linear(feat_dim, feat_dim * 2)
        self.out_emb = nn.Linear(feat_dim, feat_dim)

    def forward(self, motion: Tensor, scene_feats: Tensor) -> Tensor:
        batch_size, seq_len, _ = motion.shape
        num_points = scene_feats.shape[1]

        query = self.motion_q(motion) * (self.feat_dim ** -0.5)
        key, value = torch.split(self.scene_kv(scene_feats), [self.feat_dim, self.feat_dim], dim=-1)

        query = query.view(batch_size, self.num_head, seq_len, -1)
        key = key.view(batch_size, num_points, self.num_head, -1).transpose(1, 2)
        value = value.view(batch_size, num_points, self.num_head, -1).transpose(1, 2)

        attention = torch.matmul(query, key.transpose(2, 3))
        attention = F.softmax(attention, dim=-1)
        aggregated = torch.matmul(attention, value).view(batch_size, seq_len, -1)

        return self.out_emb(aggregated)


class SCABlock(nn.Module):
    def __init__(self, feat_dim: int, num_head: int = 4) -> None:
        super().__init__()
        self.motion_encoder = SelfAttentionLayer(num_heads=8, num_q_channels=feat_dim, dropout=0.0)
        self.sa_attention = SemanticAwareAttention(feat_dim, num_head)
        self.mlp = Mlp(feat_dim * 2, feat_dim, drop=0.0)

    def forward(self, motion: Tensor, scene_feats: Tensor, spatial_prior: Tensor) -> Tensor:
        encoded_motion = self.motion_encoder(motion)
        scene_motion_feats = self.sa_attention(encoded_motion, scene_feats, spatial_prior)
        fused = torch.cat([encoded_motion, scene_motion_feats], dim=-1)
        return self.mlp(fused)


class SemanticAwareAttention(nn.Module):
    def __init__(self, feat_dim: int, num_head: int = 6) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.motion_q = nn.Linear(feat_dim, feat_dim)
        self.scene_kv = nn.Linear(feat_dim, feat_dim * 2)
        self.out_emb = nn.Linear(feat_dim, feat_dim)

    def forward(self, motion: Tensor, scene_feats: Tensor, spatial_prior: Tensor) -> Tensor:
        batch_size, seq_len, _ = motion.shape
        num_points = scene_feats.shape[1]

        query = self.motion_q(motion) * (self.feat_dim ** -0.5)
        key, value = torch.split(self.scene_kv(scene_feats), [self.feat_dim, self.feat_dim], dim=-1)

        query = query.view(batch_size, self.num_head, seq_len, -1)
        key = key.view(batch_size, num_points, self.num_head, -1).transpose(1, 2)
        value = value.view(batch_size, num_points, self.num_head, -1).transpose(1, 2)

        attn_logits = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.feat_dim)
        attn_logits = attn_logits + spatial_prior[:, None, None, :]
        attention = F.softmax(attn_logits, dim=-1)

        aggregated = torch.matmul(attention, value).transpose(1, 2).contiguous()
        aggregated = aggregated.view(batch_size, seq_len, -1)
        return self.out_emb(aggregated)


class SpatialPrior(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, xyz: Tensor) -> Tensor:
        hidden = F.relu(self.fc1(xyz))
        return self.fc2(hidden).squeeze(-1)

@Model.register()
class TrajPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        self.pointnet = PointNet2SemSegSSGShape({'feat_dim': config.scene_feats_dim})
        self.motion_emb = nn.Linear(69 + 38, config.motion_hidden_dim)         # joints (23*3) and smpl latents (3+3+32)
        self.motion_encoder = PerceiveEncoder(n_input_channels=config.motion_hidden_dim,
                                            n_latent=config.output_seq_len + config.input_seq_len,
                                            n_latent_channels=config.motion_latent_dim,
                                            n_self_att_heads=config.motion_n_heads,
                                            n_self_att_layers=config.motion_n_layers,
                                            dropout=config.dropout)

        self.gaze_fp_layer = MyFPModule()
        self.gaze_encoder = PerceiveEncoder(n_input_channels=config.scene_feats_dim,
                                            n_latent=config.output_seq_len + config.input_seq_len,
                                            n_latent_channels=config.gaze_hidden_dim,
                                            n_self_att_heads=config.motion_n_heads,
                                            n_self_att_layers=config.gaze_n_layers,
                                            dropout=config.dropout)
        if self.config.occ:
            self.occupancy_emb = Mlp(128*128, config.motion_hidden_dim, drop=0.2)

        self.tia_blocks = nn.ModuleList([TIABlock(config.motion_latent_dim, num_head=4) for _ in range(1)])
        self.trajectory_fc = nn.Linear(config.motion_latent_dim, 6)

    def forward(self, motions, joints, scene_xyz, gazes,occupancy_map):
        """
        :param motions: (bs, seq_len, motion_dim)       [ori, trans, latent]
        :param scene_xyz: (bs, n, 3)
        :param gazes: (bs, seq_len, 1, 3)
        :return:
        """
        bs, seq_len, gaze_n, _ = gazes.shape
        assert gaze_n == 1
        gazes = gazes.squeeze(2)

        scene_feats, scene_global_feats = self.pointnet(scene_xyz.repeat(1, 1, 2))      # B x 256 x 30w, B x 256
        scene_feats = scene_feats.transpose(1, 2)
        
        motions = torch.cat([motions, joints.view(bs, seq_len, -1)], dim=-1)

        if self.config.occ:
            motions = self.motion_emb(motions) + self.occupancy_emb(occupancy_map.view(bs, -1)).unsqueeze(1)
        else:
            motions = self.motion_emb(motions)

        motions = self.motion_encoder(motions)
        gaze_embedding = self.gaze_fp_layer(gazes, scene_xyz, scene_feats.transpose(1, 2).contiguous()).transpose(1, 2)
        gaze_embedding = self.gaze_encoder(gaze_embedding)          # bs * len * c
        
        out = motions.clone()
        for blk in self.tia_blocks:
            out = blk(out, scene_xyz, scene_feats, gaze_embedding)
        out_tia = out.clone()
        ori, trans = torch.split(self.trajectory_fc(out_tia), [3, 3], dim=-1)
        
        return ori, trans
    
# class TrajPredictor(nn.Module):
#     def __init__(self, config) -> None:
#         super().__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.config = config
        
#         self.pointnet = PointNet2SemSegSSGShape({"feat_dim": config.scene_feats_dim})
#         self.motion_process = InputProcess(
#             69 + 38, config.motion_latent_dim
#         )
#         self.sequence_pos_encoder = PositionalEncoding(config.motion_latent_dim, config.dropout)
        
#         seqTransEncoderLayer = nn.TransformerEncoderLayer(
#             d_model=config.motion_latent_dim,        #256
#             nhead=config.num_heads,       #4
#             dim_feedforward=config.motion_hidden_dim,       #1024
#             dropout=config.dropout,
#             activation=config.activation,
#         )
#         self.gazeEncoder = nn.TransformerEncoder(
#             seqTransEncoderLayer, num_layers=config.motion_n_layers        # 4
#         )
        
#         self.gaze_fp_layer = MyFPModule()

#         self.occupancy_emb = Mlp(128 * 128, config.motion_hidden_dim, drop=0.1)
#         self.tia_blocks = nn.ModuleList([TIABlock(config.motion_latent_dim, num_head=4) for _ in range(1)])
#         self.trajectory_fc = nn.Linear(config.motion_latent_dim, 6)
    
#     def forward(
#         self,
#         motions: Tensor,
#         joints: Tensor,
#         scene_xyz: Tensor,
#         gazes: Tensor,
#         occupancy_map: Tensor,
#     ) -> Tuple[Tensor, Tensor]:
#         batch_size, seq_len, gaze_count, _ = gazes.shape
#         if gaze_count != 1:
#             raise ValueError(f"Expected a single gaze point per frame, got {gaze_count}.")
#         gazes = gazes.squeeze(2)

#         scene_feats, _ = self.pointnet(scene_xyz.repeat(1, 1, 2))
#         scene_feats = scene_feats.transpose(1, 2).contiguous() + self.occupancy_emb(occupancy_map.view(batch_size, -1)).unsqueeze(1)

#         motion_features = self._encode_motion(motions, joints, occupancy_map)
#         gaze_embedding = self._encode_gaze(gazes, scene_xyz, scene_feats)
#         fused_motion = self._apply_tia_blocks(motion_features, scene_xyz, scene_feats, gaze_embedding)

#         ori, trans = torch.split(self.trajectory_fc(fused_motion), [3, 3], dim=-1)

#         return ori, trans

#     def _encode_motion(self, motions: Tensor, joints: Tensor, occupancy_map: Tensor) -> Tensor:
#         batch_size, seq_len, _, _ = joints.shape
#         motion_with_joints = torch.cat([motions, joints.view(batch_size, seq_len, -1)], dim=-1)
#         motion_with_joints_copy = motion_with_joints[:,-1:].repeat(1,10,1)
#         motion_with_joints = torch.cat([motion_with_joints, motion_with_joints_copy], dim=1)
#         if self.config.occ:
#             motion_tokens = self.motion_process(motion_with_joints) + self.occupancy_emb(occupancy_map.view(batch_size, -1)).unsqueeze(1)
#         else:
#             motion_tokens = self.motion_process(motion_with_joints)
       
#         return self.sequence_pos_encoder(motion_tokens).transpose(0,1)

#     def _encode_gaze(self, gazes: Tensor, scene_xyz: Tensor, scene_feats: Tensor) -> Tensor:
#         gaze_embedding = self.gaze_fp_layer(gazes, scene_xyz, scene_feats.transpose(1, 2).contiguous())
#         gaze_embedding = self.gazeEncoder(gaze_embedding.transpose(1, 2))
#         x = gaze_embedding.permute(0, 2, 1)  # 形状变为 (8, 128, 6)
#         x_expanded = F.interpolate(x, size=16, mode='linear', align_corners=True)
#         return x_expanded.permute(0, 2, 1)
    
#     def _apply_tia_blocks(
#         self,
#         motion: Tensor,
#         scene_xyz: Tensor,
#         scene_feats: Tensor,
#         gaze_embedding: Tensor,
#     ) -> Tensor:
#         out = motion.clone()
#         for block in self.tia_blocks:
#             out = block(out, scene_xyz, scene_feats, gaze_embedding)
#         return out


@dataclass
class MotionState:
    sequence: Tensor
    current_sequence: Optional[Tensor] = None
    latent: Optional[Tensor] = None
    joints: Optional[Tensor] = None

    def __init__(self, sequence: Tensor, output_len: Optional[int] = None) -> None:
        b, t, _ = sequence.shape
        self.sequence = sequence
        motion_extend = self.sequence[:, [-1]].repeat(1, output_len, 1)
        self.sequence = torch.cat([self.sequence, motion_extend], dim=1)
        self.current_sequence = None
        self.latent = None
        self.joints = None
        
    def detach_sequence(self) -> None:
        self.sequence = self.sequence.contiguous()

@Model.register()
class ProgressivePredictor(nn.Module):
    def __init__(self, config, vposer, smplx_model) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.input_seq_len = config.input_seq_len
        self.output_seq_len = config.output_seq_len
        self.out_unit = 2
        self.use_window = True

        self.vposer = vposer
        self.smplx_model = smplx_model

        self.extractor = HumanGazeSceneUpSample(config)
        self.pointnet = PointNet2SemSegSSGShape({"feat_dim": config.scene_feats_dim})
        self.motion_emb = nn.Linear(69 + 38, config.motion_hidden_dim)
        self.motion_encoder = PerceiveEncoder(
            n_input_channels=config.motion_hidden_dim,
            n_latent=8,
            n_latent_channels=config.motion_latent_dim,
            n_self_att_heads=config.motion_n_heads,
            n_self_att_layers=config.motion_n_layers,
            dropout=config.dropout,
        )

        self.spatial_att = SpatialPrior()
        self.dist_factor = nn.Parameter(torch.zeros(1))
        self.dir_factor = nn.Parameter(torch.zeros(1))

        self.traj_predictor = TrajPredictor(config)
        self.load_ckpts()

        self.sca_blocks = nn.ModuleList([SCABlock(config.motion_latent_dim, num_head=4) for _ in range(1)])
        self.pose_fc = nn.Linear(config.motion_latent_dim, 32)

        self.joints_fc = nn.Linear(config.motion_latent_dim * 2, 69)
        self.motion_decoder = GCN(config, node_n=69)
        self.motion_decoder2 = GCN(config, node_n=69)

    def load_ckpts(self) -> None:
        ckpts= "/data/wuyang/output/traj_ckpts/best_128.pth"
        state_dict = torch.load(ckpts, map_location=self.device)
        self.traj_predictor.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        motions: Tensor,
        joints: Tensor,
        scene_xyz: Tensor,
        gazes: Tensor,
        occupancy: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, _, _, _ = joints.shape
        with torch.no_grad():
            ori, trans = self.traj_predictor(motions, joints, scene_xyz, gazes, occupancy)

        human_points = self.extractor.cut_by_step(scene_xyz, trans)     # [B, output_seq_len/2, num_points, 3]

        scene_feats_batch = self._extract_scene_features(human_points.view(batch_size * self.output_seq_len // self.out_unit, -1, 3))        # [B, num_points, feat_dim]
        scene_feats_batch = scene_feats_batch.view(batch_size, self.output_seq_len // self.out_unit, -1, self.config.scene_feats_dim)

        motion_state = MotionState(
                sequence=torch.cat([motions, joints.view(batch_size, self.input_seq_len, -1)], dim=-1),
                output_len=self.output_seq_len
            )
        
        for step in range(self.output_seq_len // self.out_unit):
            scene_feats = scene_feats_batch[:, step]
            motion_state, start, end = self._update_motion_sequence(motion_state, step)
            
            encoded_motion = self._encode_motion_sequence(motion_state.current_sequence)
            spatial_prior = self._compute_spatial_prior(human_points[:, step])
            fused_motion = self._apply_sca_blocks(encoded_motion, scene_feats, spatial_prior)          
            
            pose_tokens = self.pose_fc(fused_motion)
            motion_state = self._accumulate_predictions(
                motion_state, pose_tokens, ori[:, start:end], trans[:, start:end], joints
            )
            
        reconstructed = latent_to_joints(motion_state.latent, self.vposer, self.smplx_model)[:, :, :23]
        reconstructed = torch.cat([joints, reconstructed[:, 6:]], dim=1).clone()
        pred_joints = reconstructed[:, :, :23]


        pred_joints = self.motion_decoder(pred_joints)
        pred_joints = self.pose_traj_fusion(trans.unsqueeze(2), pred_joints)
        pred_joints = self.motion_decoder2(pred_joints)
        return motion_state.latent, pred_joints

    def pose_traj_fusion(self, traj: Tensor, joints: Tensor) -> Tensor:
        """
        Anchor predicted joints to the forecasted trajectory, keeping the pelvis aligned.
        """
        predictions = joints.clone()
        distance = predictions[:, :, [0], :] - traj
        predictions = predictions - distance
        predictions[:, :, [0], :] = traj
        return predictions

    # TODO: 使用mlp去替代，减少性能消耗
    def _encode_motion_sequence(self, motions: Tensor) -> Tensor:
        motion_tokens = self.motion_emb(motions)
        return self.motion_encoder(motion_tokens)

    def _update_motion_sequence(self, state: MotionState, step: int) -> Tuple[MotionState, int, int]:
        # 多轮次的预测优化
        if self.use_window:
            start = step * self.out_unit
            end = start + self.input_seq_len + self.out_unit
            
            state.current_sequence = state.sequence[:, start:end]
            return state, start, end
        
        # TODO: 还未确定是否可行
        if state.latent is None or state.joints is None:
            raise RuntimeError("Sliding window disabled requires previous latent predictions.")

        start = 0
        end = self.input_seq_len + step * self.out_unit
        new_motion = torch.cat(
            [
                state.latent[:, -self.out_unit :],
                state.joints[:, -self.out_unit :].view(state.joints.size(0), self.out_unit, -1),
            ],
            dim=-1,
        )
        state.sequence = torch.cat([state.sequence, new_motion], dim=1)
        return state, start, end

    def _extract_scene_features(self, scene_points: Tensor) -> Tensor:
        scene_feats, _ = self.pointnet(scene_points.repeat(1, 1, 2))
        return scene_feats.transpose(1, 2).contiguous()

    def _compute_spatial_prior(self, scene_points: Tensor) -> Tensor:
        distance = torch.norm(scene_points, dim=-1)
        distance_salience = -((distance * 1.5) ** 3)
        direction_salience = scene_points[:, :, 0] / torch.sqrt((scene_points[:, :, [0, 2]] ** 2).sum(dim=-1))
        base_prior = self.spatial_att(scene_points)
        return distance_salience * self.dist_factor + direction_salience * self.dir_factor + base_prior

    def _apply_sca_blocks(self, motion: Tensor, scene_feats: Tensor, spatial_prior: Tensor) -> Tensor:
        out = motion.clone()
        for block in self.sca_blocks:
            out = block(out, scene_feats, spatial_prior)
        return out.clone()

    def _accumulate_predictions(
        self,
        state: MotionState,
        pose_tokens: Tensor,
        ori_slice: Tensor,
        trans_slice: Tensor,
        input_joints: Tensor,
    ) -> MotionState:
        latent = torch.cat([ori_slice, trans_slice, pose_tokens], dim=-1)

        if state.latent is None:
            latent_full = latent
        else:
            latent_full = torch.cat([state.latent, latent[:, -self.out_unit :]], dim=1)

        state.latent = latent_full
        return state

