import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pointnet_plus2 import PointNet2SemSegSSGShape, MyFPModule
from model.sif3d_traj import TrajPredictor
from model.partial_scene import EgoSceneExtractor
from model.base_cross_model import GatedSelfAttention
import math
from utils.vis_utils import latent_to_joints, visualize_SMPLXjoints, visualize_points, visualize_Scene_wo_color
from model.base_cross_model import PerceiveEncoder

# implement 2 for this task

class RollingPose(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_seq_len = config.input_seq_len
        self.output_seq_len = config.output_seq_len
        self.out_unit = config.out_unit
        self.time_emb_dim = config.motion_latent_dim

        self.traj_predictor = TrajPredictor(config)
        self.pointnet = PointNet2SemSegSSGShape({'feat_dim': config.scene_feats_dim})
        self.partial_fp_layer = MyFPModule()
        self.extractor = EgoSceneExtractor(config)
        
        self.input_process = InputProcess(
            38, config.motion_latent_dim
        )
        self.motion_encoder = PerceiveEncoder(n_input_channels=config.motion_hidden_dim,
                                        n_latent=self.input_seq_len+self.out_unit,   # T seq_len
                                        n_latent_channels=config.motion_latent_dim,
                                        n_self_att_heads=config.motion_n_heads,
                                        n_self_att_layers=config.motion_n_layers,
                                        dropout=config.dropout)
        # self.sequence_pos_encoder = PositionalEncoding(config.motion_latent_dim, config.dropout)
        self.sequence_pos_encoder = RotaryPositionalEmbedding(config.motion_latent_dim)

        self.spatial_att = SpatialPrior()
        self.dist_factor = nn.Parameter(torch.zeros(1))
        self.dir_factor = nn.Parameter(torch.zeros(1))
        
        self.sca_blocks = nn.ModuleList([SCABlock(config.motion_latent_dim, num_head=4) for _ in range(2)])
        # self.ot_fc = nn.Linear(config.motion_latent_dim, 6)
        self.pose_fc = nn.Linear(config.motion_latent_dim, 32)

    def _update_motion_sequence(self, motions, trans, step):
        start = step * self.out_unit
        end = start + self.input_seq_len + self.out_unit
        return motions[:, start:end], trans[:,start:end], start, end-self.out_unit-1, end
    
    def _compute_spatial_prior(self, scene_points):
        distance = torch.norm(scene_points, dim=-1)
        distance_salience = -((distance * 1.5) ** 3)
        direction_salience = scene_points[:, :, 0] / torch.sqrt((scene_points[:, :, [0, 2]] ** 2).sum(dim=-1))
        base_prior = self.spatial_att(scene_points)
        return distance_salience * self.dist_factor + direction_salience * self.dir_factor + base_prior
    
    def _apply_sca_blocks(self, motion, scene_feats, spatial_prior):
        out = motion.clone()
        for block in self.sca_blocks:
            out = block(out, scene_feats, spatial_prior)
        return out.clone()
    
    def forward(self, motions, joints, scene_xyz, scene_feats, ori, trans):
        """
            :param motions: (bs, seq_len, motion_dim)       [ori, trans, latent] dim:38
            :param joints: (bs, his_len, 23, 3)
            :param scene_xyz: (bs, n, 3)
            :param gazes: (bs, seq_len, 1, 3)
            :param occ: (bs, 128, 128)
            :return:
        """        # motion_extend = torch.zeros(B, self.output_seq_len, 38)
        # TODO: 修改这个extend
        motion_extend = motions[:, [-1]].repeat(1, self.output_seq_len, 1)
        motions_cat = torch.cat([motions, motion_extend], dim=1)        #B，16，38
        # change motion to fixed trans
        motions_cat = torch.cat([ori,trans,motions_cat[:,:,6:]], dim=-1)   # B,_,38

        latent_full = motions_cat.clone()   # B,T,38
        visualize_Scene_wo_color(scene_xyz[0])
        for step in range(self.output_seq_len //self.out_unit):
            # 截取一个 window 大小的 motion  0，8
            # TODO：在这个里可以将上面的extend给融合到这个方法里面，每次重新填充
            latent, trans_window, start, cur_index, end = self._update_motion_sequence(latent_full, trans, step)
            # 进行encode
            motions = self.input_process(latent).transpose(0,1)  # --> [seqlen, bs, d]
            motions = self.motion_encoder(motions)  # [seqlen, bs, d]
            # TODO:场景基点存在问题，pose取的应该是倒数第3个，测试两种方法：
            # 1、当前帧与前一个计算瞬时速度；
            local_scene =  self.extractor(scene_xyz, trans_window[:, -self.out_unit - 2:-self.out_unit], ori[:, cur_index])  # B,nh,3
            
            # 2、根据已知的trans直接计算 unit 内的速度，但是提取场景还是当前帧
            # local_scene =  self.extractor(scene_xyz, trans_window[:,[cur_index,-1]], ori[:, cur_index])  # B,nh,3
            visualize_Scene_wo_color(local_scene[0], "sampled_scene", step)

            egoscene_feats = self.partial_fp_layer(local_scene, scene_xyz, scene_feats.transpose(1,2).contiguous()).transpose(1, 2).contiguous()
            
            # B,1024
            spatial_prior = self._compute_spatial_prior(local_scene)
            out_sca = self._apply_sca_blocks(motions, egoscene_feats, spatial_prior)
            
            # pred_ori, pred_trans = self.ot_fc(out_sca).chunk(2, dim=-1)  # B,T,3 , B,T,3
            pose = self.pose_fc(out_sca) # B,T,32

            # trans_window = trans_window + torch.tanh(trans_window - pred_trans)
            
            latent_pred = torch.cat([ori[:, start:end], trans[:, start:end], pose], dim=-1) 
            # trans = torch.cat([trans[:,:end-self.out_unit], trans_window[:, -self.out_unit :],trans[:,end:]], dim=1)
            latent_full = torch.cat([latent_full[:,:end-self.out_unit], latent_pred[:, -self.out_unit :],latent_full[:,end:]], dim=1)

        return latent_full

class TransformerFilling(nn.Module):
    """
    A Transformer used for filling predictions in sequences with masks.
    This is demonstrated using PyTorch's built-in nn.TransformerEncoder/Decoder.
    """
    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,dropout=0.3)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (T, B, D)  # Source sequence with shape (T, B, D)
        tgt: (T, B, D)  # Target sequence with shape (T, B, D)
        src_mask: Mask for the source sequence (optional)
        tgt_mask: Mask for the target sequence (optional)
        """
        memory = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        out = self.transformer_decoder(tgt, memory, tgt_mask, src_mask)
        return out


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

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        # Initialize cache
        self._update_cache(max_position_embeddings)

    def _update_cache(self, max_seq_len):
        self.max_seq_len_cached = max_seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        # [seq_len, 1, dim] for broadcasting to [seq_len, bs, dim]
        self.register_buffer("cos_cached", emb.cos()[:, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[:, None, :], persistent=False)

    def _apply_rotary_pos_emb(self, x, cos, sin):
        # x: [seq_len, bs, dim]
        # rotate_half
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        
        return (x * cos) + (x_rotated * sin)
    
    def forward(self, x):
        # x: [seq_len, bs, dim]
        seq_len = x.shape[0]
        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len)
            
        cos = self.cos_cached[:seq_len, ...].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:seq_len, ...].to(dtype=x.dtype, device=x.device)
        
        return self._apply_rotary_pos_emb(x, cos, sin)

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


class SCABlock(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.motion_encoder = GatedSelfAttention(num_heads=8, num_q_channels=feat_dim, dropout=0.0)
        self.sa_attention = SemanticAwareAttention(feat_dim, num_head)
        self.mlp = Mlp(feat_dim * 2, feat_dim, drop=0.0)

    def forward(self, motion, scene_feats, spatial_prior):
        """
        motion: B,T,256
        scene_feats: B,N,256
        spatial_prior: B,N
        """
        motion_sca = self.motion_encoder(motion)    # B，T，256
        sm_feature = self.sa_attention(motion_sca, scene_feats, spatial_prior)  # B，T，256
        out = torch.cat([motion_sca, sm_feature], dim=-1)   # B，T，512
        out = self.mlp(out) #B,T,256
        return out


class SemanticAwareAttention(nn.Module):
    def __init__(self, feat_dim, num_head=6):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.motion_q = nn.Linear(feat_dim, feat_dim)
        self.scene_kv = nn.Linear(feat_dim, feat_dim * 2)
        self.out_emb = nn.Linear(feat_dim, feat_dim)

    def forward(self, motion, scene_feats, spatial_prior):
        bs, len, _ = motion.shape
        N = scene_feats.shape[1]

        q = self.motion_q(motion)           # bs * len * c
        q = q * (self.feat_dim ** -0.5)
        k, v = torch.split(self.scene_kv(scene_feats), [self.feat_dim, self.feat_dim], dim=-1)
        # bs * head * len * _c, bs * head * N * _c
        q, k, v = q.reshape(bs, self.num_head, len, -1), k.reshape(bs, N, self.num_head, -1).transpose(1, 2), v.reshape(bs, N, self.num_head, -1).transpose(1, 2)

        #TODO: scaled dot-product attention
        att = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.feat_dim)       # bs * head * len * N
        att = att + spatial_prior[:, None, None, :]
        att = F.softmax(att, dim=-1)                         # bs * head * len * N
        out = torch.matmul(att, v).transpose(1, 2).contiguous().view(bs, len, -1)           # bs, len, c
        out = self.out_emb(out)

        return out


class SpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, xyz):
        out = F.gelu(self.fc1(xyz))
        spatial_prior = self.fc2(out)
        return spatial_prior.squeeze(-1)

class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=4, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim * expansion)
        self.fc2 = nn.Linear(out_dim * expansion, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out = self.drop(F.gelu(self.fc1(x)))
        out = self.drop(self.fc2(out))
        return out
