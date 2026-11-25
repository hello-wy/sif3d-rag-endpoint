import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GCN
from utils.vis_utils import latent_to_joints,visualize_Scene_wo_color
from model.base_cross_model import PerceiveEncoder
from model.pointnet_plus2 import PointNet2SemSegSSGShape, MyFPModule
from model.base_cross_model import SelfAttentionLayer
from model.partial_scene import HumanGazeSceneUpSample
import math
import hashlib
from pathlib import Path
from model.base_model import Model
from utils.rotation import axis_angle_to_matrix, matrix_to_axis_angle
from model.multimodal_rag import (
    MotionBank,
    MotionRefiner,
    LocalMotionRefiner,
    PrototypeBlendRefiner,
    EndpointPrototypeRefiner,
    TerminalBlendRefiner,
    AdaptiveTerminalRefiner,
    endpoint_candidates,
    reference_frame,
    to_reference,
    forecast_candidate_scores,
    MultiModalRetriever,
    PrototypeConditionAdapter,
    PrototypeEncoder,
)


def human_centered_scene(xyz, ori, trans):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        ori: input global orient of the human, [B, len, 3]
        trans: input translation of the human, [B, len, 3]
    Return:
        new_xyz: normalized points position data, [B, len, N, 3]
    """
    bs, T, _ = ori.shape
    sin_ori = torch.sin(ori[:, :, 1]).unsqueeze(-1)
    cos_ori = torch.cos(ori[:, :, 1]).unsqueeze(-1)
    xyz = xyz[:, None, :, :].repeat(1, T, 1, 1) - trans[:, :, None, :]        # [B, len, N, 3]

    # First column is the X-axis transform, second for Y, and the third column for Z
    M = torch.cat([torch.cat([sin_ori, torch.zeros_like(sin_ori), -cos_ori], dim=-1),
                   torch.cat([torch.zeros_like(sin_ori), torch.ones_like(sin_ori), torch.zeros_like(sin_ori)], dim=-1),
                   torch.cat([cos_ori, torch.zeros_like(sin_ori), sin_ori], dim=-1)], dim=-1)
    M = M.view(bs, T, 3, 3)
    new_xyz = torch.matmul(xyz, M)

    return new_xyz


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

@Model.register()
class SIF3D(nn.Module):
    def __init__(self, config, vposer, smplx_model):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.input_seq_len = config.input_seq_len
        self.output_seq_len = config.output_seq_len
        self.out_unit = config.out_unit
        self.use_window = True
        
        self.vposer = vposer
        self.smplx_model = smplx_model

        self.extractor = HumanGazeSceneUpSample(config)
        self.pointnet = PointNet2SemSegSSGShape({'feat_dim': config.scene_feats_dim})
        
        self.motion_emb = nn.Linear( 38, config.motion_hidden_dim)         # joints (23*3) and smpl latents (3+3+32)
        self.motion_encoder = PerceiveEncoder(n_input_channels=config.motion_hidden_dim,
                                              n_latent=self.input_seq_len+self.out_unit,   # T seq_len
                                              n_latent_channels=config.motion_latent_dim,
                                              n_self_att_heads=config.motion_n_heads,
                                              n_self_att_layers=config.motion_n_layers,
                                              dropout=config.dropout,
                                              use_rotary_pos_emb=config.rope)

        
        self.spatial_att = SpatialPrior()
        self.dist_factor = nn.Parameter(torch.zeros(1))
        self.dir_factor = nn.Parameter(torch.zeros(1))

        self.traj_predictor = TrajPredictor(config)
        self.load_ckpts()
        
        self.sca_blocks = nn.ModuleList([SCABlock(config.motion_latent_dim, num_head=4) for _ in range(1)])
        self.pose_fc = nn.Linear(config.motion_latent_dim, 32)

        self.motion_decoder = GCN(config, node_n=69)

        self.use_rag = bool(getattr(config, 'use_rag', False))
        self.use_refiner = (bool(getattr(config, 'use_refiner', False)) or self.use_rag) and not bool(getattr(config, 'rag_no_refiner', False))
        self.rag_bank = None
        self.pose_bank = False
        self.endpoint_adapter = bool(getattr(config, 'rag_endpoint_adapter', False))
        self.rag_retriever = None
        self.prototype_encoder = None
        self.prototype_adapter = None
        self.motion_refiner = None
        rag_dim = int(getattr(config, 'rag_embedding_dim', config.motion_latent_dim))
        if self.use_rag:
            bank_path = getattr(config, 'rag_bank', None) or getattr(config, 'rag_index_path', None)
            if not bank_path:
                raise ValueError('--use_rag requires --rag_bank')
            self.rag_bank = MotionBank.load(bank_path, map_location='cpu')
            from model.endpoint_pose_bank import EndpointPoseBank
            self.pose_bank = isinstance(self.rag_bank, EndpointPoseBank)
            if self.pose_bank != (config.rag_refiner_type in ('pose_endpoint', 'pose_endpoint_temporal', 'pose_endpoint_aligned', 'pose_endpoint_trust', 'pose_endpoint_joint_trust', 'pose_endpoint_interpolation', 'pose_endpoint_terminal', 'pose_endpoint_local')):
                raise ValueError('Endpoint pose banks require a pose_endpoint refiner')
            if self.endpoint_adapter and not self.pose_bank:
                raise ValueError('Endpoint adapter requires a single-frame pose bank')
            if not self.rag_bank.manifest.get('retriever_trained', False):
                raise ValueError('Train the retriever and rebuild keys before enabling RAG')
            for key in ('input_seq_len', 'output_seq_len', 'fps', 'sample_points'):
                if self.rag_bank.manifest.get(key) != getattr(config, key):
                    raise ValueError(f'MotionBank/config {key} mismatch; rebuild a matching bank')
            if self.rag_bank.manifest.get('key_hash') != self.rag_bank.encoder_hash():
                raise ValueError('MotionBank key hash mismatch')
            self.register_buffer('rag_bank_signature', torch.tensor(list(hashlib.sha256(Path(bank_path).read_bytes()).digest()), dtype=torch.uint8))
            modalities = str(getattr(config, 'rag_modalities', 'motion+gaze+scene'))
            use_motion = 'motion' in modalities
            use_gaze = 'gaze' in modalities
            use_scene = 'scene' in modalities
            # Legacy switches remain valid aliases for older launch files.
            if bool(getattr(config, 'rag_motion_only', False)):
                use_motion, use_gaze, use_scene = True, False, False
            elif bool(getattr(config, 'rag_motion_scene', False)):
                use_motion, use_gaze, use_scene = True, False, True
            elif bool(getattr(config, 'rag_motion_gaze', False)):
                use_motion, use_gaze, use_scene = True, True, False
            self.rag_retriever = MultiModalRetriever(
                motion_dim=38,
                joint_count=23,
                embedding_dim=rag_dim,
                use_motion=use_motion,
                use_gaze=use_gaze,
                use_scene=use_scene,
                dropout=0.0,
                temporal=self.rag_bank.manifest.get('encoder_version') == 'multimodal-retriever-v2',
                rich_summary=self.rag_bank.manifest.get('encoder_version') == 'multimodal-retriever-v3',
                scene_focus=self.rag_bank.manifest.get('encoder_version') in ('multimodal-retriever-v4', 'multimodal-retriever-v5', 'multimodal-retriever-v6'),
                gaze_heading=self.rag_bank.manifest.get('encoder_version') == 'multimodal-retriever-v4',
                scene_hist=self.rag_bank.manifest.get('encoder_version') == 'multimodal-retriever-v6',
            )
            self.prototype_encoder = PrototypeEncoder(23, rag_dim)
            self.prototype_adapter = PrototypeConditionAdapter(
                config.motion_latent_dim,
                rag_dim,
                heads=4,
                dropout=float(getattr(config, 'rag_prototype_dropout', 0.0)),
            )
            if self.use_rag:
                bank_path = getattr(config, 'rag_bank', None) or getattr(config, 'rag_index_path', None)
                if not bank_path:
                    raise ValueError('--use_rag requires --rag_bank pointing to a frozen train-only MotionBank')
                if self.rag_bank.keys is None:
                    raise ValueError('MotionBank has no retrieval keys; build keys with the matching frozen encoder first')
                if self.rag_bank.encoder_state_dict is None:
                    raise ValueError('MotionBank has no encoder checkpoint; rebuild the bank instead of querying an unknown key space')
                state = self.rag_bank.encoder_state_dict
                external_ckpt = getattr(config, 'rag_retriever_ckpt', None)
                if external_ckpt:
                    external = torch.load(external_ckpt, map_location='cpu')
                    external = external.get('state_dict', external)
                    if set(external) != set(state) or any(not torch.equal(external[k], state[k]) for k in state):
                        raise ValueError('retriever checkpoint does not match the MotionBank key encoder; rebuild the index')
                    state = external
                self.rag_retriever.load_state_dict(state, strict=True)
                bank_flags = {k: self.rag_bank.manifest[k] for k in ('use_motion', 'use_gaze', 'use_scene') if k in self.rag_bank.manifest}
                current_flags = {
                    'use_motion': self.rag_retriever.use_motion,
                    'use_gaze': self.rag_retriever.use_gaze,
                    'use_scene': self.rag_retriever.use_scene,
                }
                if bank_flags and bank_flags != current_flags:
                    raise ValueError(f'MotionBank modality flags {bank_flags} != config {current_flags}; rebuild/query with the matching encoder')
                if self.rag_bank.manifest.get('encoder_version', self.rag_retriever.encoder_version) != self.rag_retriever.encoder_version:
                    raise ValueError('MotionBank encoder version does not match the active retriever; rebuild the index')
                expected = self.rag_retriever.embedding_dim
                if self.rag_bank.keys.shape[-1] != expected:
                    raise ValueError(f'MotionBank embedding dim {self.rag_bank.keys.shape[-1]} != retriever dim {expected}; rebuild the index')
                for p in self.rag_retriever.parameters():
                    p.requires_grad_(False)
                self.rag_retriever.eval()
        if self.use_refiner:
            if config.rag_refiner_type in ('blend', 'endpoint', 'terminal_blend','adaptive_terminal',
                                           'pose_endpoint', 'pose_endpoint_temporal', 'pose_endpoint_aligned', 'pose_endpoint_trust', 'pose_endpoint_joint_trust', 'pose_endpoint_interpolation', 'pose_endpoint_terminal', 'pose_endpoint_local') and not self.use_rag:
                raise ValueError('Prototype blending requires --use_rag and a train-only bank')
            from model.endpoint_pose_bank import EndpointPoseRefiner, EndpointPoseTemporalRefiner, EndpointTrustRefiner, EndpointJointTrustRefiner, EndpointInterpolationRefiner, EndpointTerminalRefiner, EndpointLocalMotionRefiner
            refiner_class = {'local': LocalMotionRefiner, 'legacy': MotionRefiner, 'blend': PrototypeBlendRefiner, 'endpoint': EndpointPrototypeRefiner, 'terminal_blend': TerminalBlendRefiner,'adaptive_terminal':AdaptiveTerminalRefiner, 'pose_endpoint': EndpointPoseRefiner, 'pose_endpoint_temporal': EndpointPoseTemporalRefiner, 'pose_endpoint_aligned': EndpointPoseRefiner, 'pose_endpoint_trust': EndpointTrustRefiner, 'pose_endpoint_joint_trust': EndpointJointTrustRefiner, 'pose_endpoint_interpolation': EndpointInterpolationRefiner, 'pose_endpoint_terminal': EndpointTerminalRefiner, 'pose_endpoint_local': EndpointLocalMotionRefiner}[config.rag_refiner_type]
            kwargs = {'max_delta': config.rag_max_delta} if config.rag_refiner_type == 'local' else {}
            if config.rag_refiner_type in ('endpoint','terminal_blend','adaptive_terminal'):
                kwargs.update(endpoint_frames=config.rag_endpoint_frames)
                if config.rag_refiner_type == 'endpoint':kwargs.update(endpoint_topk=config.rag_endpoint_topk)
                if config.rag_refiner_type == 'adaptive_terminal':kwargs.update(joint_trust=config.rag_terminal_joint_trust,part_trust=config.rag_terminal_part_trust,max_trust=config.rag_terminal_max_trust)
            self.motion_refiner = (refiner_class(frames=config.output_seq_len) if config.rag_refiner_type in ('pose_endpoint_trust', 'pose_endpoint_joint_trust', 'pose_endpoint_interpolation', 'pose_endpoint_terminal', 'pose_endpoint_local')
                                   else refiner_class(frames=config.output_seq_len,
                                                rerank_topk=config.rag_endpoint_topk if config.rag_forecast_rerank else 0,
                                                align_yaw=config.rag_refiner_type == 'pose_endpoint_aligned')
                                   if config.rag_refiner_type in ('pose_endpoint', 'pose_endpoint_temporal', 'pose_endpoint_aligned')
                                   else refiner_class(motion_dim=38, joint_count=23, context_dim=rag_dim,
                                                      prototype_dim=rag_dim, hidden_dim=config.rag_refiner_hidden_dim, **kwargs))
        if (self.use_rag or self.use_refiner) and config.rag_train_stage in ('adapter', 'refiner'):
            self.requires_grad_(False)
            modules = (self.prototype_encoder, self.motion_refiner)
            if config.rag_train_stage == 'adapter':
                modules += (self.prototype_adapter,)
            for module in modules:
                if module is not None:
                    module.requires_grad_(True)
            if config.rag_refiner_type in ('pose_endpoint', 'pose_endpoint_temporal', 'pose_endpoint_aligned', 'pose_endpoint_trust', 'pose_endpoint_joint_trust', 'pose_endpoint_interpolation', 'pose_endpoint_terminal', 'pose_endpoint_local'):
                self.prototype_encoder.requires_grad_(self.endpoint_adapter and config.rag_train_stage == 'adapter')
            if config.rag_refiner_type in ('blend', 'endpoint', 'terminal_blend','adaptive_terminal') and self.prototype_encoder is not None:
                self.prototype_encoder.requires_grad_(False)
                if config.rag_refiner_type in ('endpoint','terminal_blend','adaptive_terminal'):
                    self.motion_refiner.mix.requires_grad_(config.rag_train_sequence_mix and config.rag_refiner_type=='endpoint')

    def train(self, mode=True):
        super().train(mode)
        if self.rag_retriever is not None:
            self.rag_retriever.eval()
        if (self.use_rag or self.use_refiner) and self.config.rag_train_stage in ('adapter', 'refiner'):
            for name, module in self.named_children():
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()
        return self

    def load_prediction_checkpoint(self, weights, allow_baseline=False):
        weights = {k: v for k, v in weights.items() if not k.startswith('smplx_model.')}
        prefixes = ('rag_', 'prototype_encoder.', 'prototype_adapter.', 'motion_refiner.')
        baseline = allow_baseline and not any(k.startswith(prefixes) for k in weights)
        if self.motion_refiner is not None and hasattr(self.motion_refiner,'trust_limit'):
            saved_limit = weights.get('motion_refiner.trust_limit')
            if not baseline and (saved_limit is None or not torch.equal(saved_limit.cpu(),self.motion_refiner.trust_limit.cpu())):
                raise ValueError('Checkpoint terminal trust limit does not match the configured refiner')
        if self.rag_bank is not None and not baseline:
            signature = weights.get('rag_bank_signature')
            if signature is None or not torch.equal(signature.cpu(), self.rag_bank_signature.cpu()):
                raise ValueError('Full checkpoint must use the same MotionBank')
            for key, value in self.rag_bank.encoder_state_dict.items():
                loaded = weights.get('rag_retriever.' + key)
                if loaded is None or not torch.equal(loaded.cpu(), value.cpu()):
                    raise ValueError('Checkpoint/key encoder mismatch; use matching rebuilt bank')
        incompatible = self.load_state_dict(weights, strict=False)
        illegal = [k for k in incompatible.missing_keys if not k.startswith('smplx_model.') and
                   not (baseline and k.startswith(prefixes))]
        if illegal or incompatible.unexpected_keys:
            raise ValueError(f'Incompatible checkpoint: missing={illegal}, unexpected={incompatible.unexpected_keys}')
        return incompatible

    def load_ckpts(self):
        state_dict = torch.load(self.config.traj_ckpts, map_location='cpu')
        print(self.config.traj_ckpts)
        self.traj_predictor.load_state_dict(state_dict, strict=False)
        
    def _extract_scene_features(self, scene_points):
        scene_feats, _ = self.pointnet(scene_points.repeat(1, 1, 2))
        return scene_feats.transpose(1, 2).contiguous()
    
    # TODO: 使用mlp去替代，减少性能消耗
    def _encode_motion_sequence(self, motions):
        motion_tokens = self.motion_emb(motions)
        return self.motion_encoder(motion_tokens)
    
    def _update_motion_sequence(self, motions, step):
        start = step * self.out_unit
        end = start + self.input_seq_len + self.out_unit
        return motions[:, start:end], start, end
    
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
    
    def combine_smplx_rt(
        self,
        trans_global:    torch.Tensor,  # (B, T, 3) 全局平移
        trans_residual:  torch.Tensor,  # (B, T, 3) 平移残差
        ori_global:      torch.Tensor,  # (B, T, 3) 全局旋转 (axis-angle, SMPLX global_orient)
        ori_residual:    torch.Tensor,  # (B, T, 3) 旋转残差 (axis-angle)
    ):
        """
        合并 SMPLX 的全局平移/旋转 与 残差，得到最终平移 & 最终旋转（axis-angle）

        返回:
            final_trans:   (B, T, 3) 最终平移 = trans_global + trans_residual
            final_orient:  (B, T, 3) 最终旋转 (axis-angle)
        """
        # 1. 平移：逐元素相加即可
        final_trans = trans_global + trans_residual       # (B, T, 3)

        # 2. 旋转：先把两个 axis-angle 转成矩阵
        Rg = axis_angle_to_matrix(ori_global)             # (B, T, 3, 3) 全局
        Rr = axis_angle_to_matrix(ori_residual)           # (B, T, 3, 3) 残差

        # 假设“最终旋转 = 残差 * 全局”
        R_final = torch.matmul(Rr, Rg)                    # (B, T, 3, 3)

        # 再从矩阵转回 axis-angle
        final_orient = matrix_to_axis_angle(R_final)      # (B, T, 3)

        return final_trans, final_orient

    def _retrieve_prototypes(self, motions, joints, scene_xyz, gazes, metadata=None):
        if self.rag_bank is None or self.rag_retriever is None:
            return None
        if metadata is None or any(not row.get('sequence_id') for row in metadata):
            raise ValueError('Retrieval requires sequence_id metadata to exclude the query recording')
        with torch.no_grad():
            query = self.rag_retriever(None if self.pose_bank else motions, joints, gazes, scene_xyz)
            result = self.rag_bank.retrieve(
                query,
                top_k=int(getattr(self.config, 'rag_top_k', 5)),
                metadata=metadata,
                exclude_same_sequence=bool(getattr(self.config, 'rag_exclude_same_sequence', True)),
                exclude_scene=bool(getattr(self.config, 'rag_exclude_scene', False)),
                chunk_size=int(getattr(self.config, 'rag_chunk_size', 4096)),
                backend=str(getattr(self.config, 'rag_backend', 'torch')),
                unique_sequences=self.config.rag_unique_sequences,
                prefer_same_scene=self.config.rag_prefer_same_scene,
            )
            if bool(getattr(self.config, 'rag_random_prototype', False)):
                for b, meta in enumerate(metadata):
                    legal = torch.where(self.rag_bank.legal_candidate_mask(meta,
                        self.config.rag_exclude_same_sequence, self.config.rag_exclude_scene))[0].tolist()
                    if self.config.rag_prefer_same_scene:
                        legal.sort(key=lambda idx: self.rag_bank.items[idx].scene_id != meta['scene_id'])
                    take = min(len(legal), self.config.rag_top_k)
                    order = torch.randperm(len(legal)).tolist()
                    if self.config.rag_prefer_same_scene:
                        order.sort(key=lambda pos: self.rag_bank.items[legal[pos]].scene_id != meta['scene_id'])
                    if self.config.rag_unique_sequences:
                        chosen, seen = [], set()
                        for position in order:
                            idx = legal[position]; sequence = self.rag_bank.items[idx].sequence_id
                            if sequence in seen:
                                continue
                            seen.add(sequence); chosen.append(idx)
                            if len(chosen) == self.config.rag_top_k:
                                break
                        take = len(chosen)
                        ids = torch.tensor(chosen, device=query.device, dtype=torch.long)
                    else:
                        ids = torch.tensor([legal[pos] for pos in order[:take]], device=query.device, dtype=torch.long)
                    result['retrieved_ids'][b].fill_(-1)
                    result['retrieval_scores'][b].fill_(-torch.inf)
                    result['candidate_valid_mask'][b].fill_(False)
                    result['retrieved_ids'][b, :take] = ids
                    result['retrieval_scores'][b, :take] = 0.0
                    result['candidate_valid_mask'][b, :take] = True
            if self.pose_bank:
                endpoint_poses = self.rag_bank.gather_pose(result['retrieved_ids'], joints)
            else:
                endpoint_poses = None
            if self.pose_bank:
                future = None
            else:
                future = self.rag_bank.gather_future(result['retrieved_ids'], joints)
                world_future = future
                if self.config.rag_refiner_type == 'local':
                    refs = [reference_frame(row) for row in joints]
                    future = torch.stack([to_reference(row, o, basis) for row, (o, basis) in zip(future, refs)])
        if self.pose_bank:
            payload = {'query': query, 'endpoint_poses': endpoint_poses, **result}
            if self.endpoint_adapter:
                observed = joints[:, -1] - joints[:, -1, :1]
                terminal = endpoint_poses - endpoint_poses[:, :, :1]
                ramp = torch.linspace(.1, 1., self.output_seq_len,
                                      device=joints.device, dtype=joints.dtype)[None, None, :, None, None]
                synthetic = (1 - ramp) * observed[:, None, None] + ramp * terminal[:, :, None]
                payload['prototype_tokens'] = self.prototype_encoder(synthetic)
            return payload
        # Keep this outside the retrieval no-grad region: PrototypeEncoder is
        # one of the stage-D modules trained against the future loss.
        prototype_tokens = self.prototype_encoder(future)
        return {
            'query': query,
            'future_joints': future,
            'world_future_joints': world_future,
            'prototype_tokens': prototype_tokens,
            **result,
        }

    def forward(self, motions, joints, scene_xyz, gazes, occ, rag_metadata=None, return_aux=False):
        """
        :param motions: (bs, seq_len, motion_dim)       [ori, trans, latent] dim:38
        :param scene_xyz: (bs, n, 3)
        :param gazes: (bs, seq_len, 1, 3)
        :return:
        """
        B, _, J, C = joints.shape
        # The rolling loop below reuses ``motions`` for encoder hidden states;
        # keep the original 38-D history for MotionRefiner's contract.
        history_motions = motions
        rag_enabled = not self.config.disable_rag
        rag = self._retrieve_prototypes(motions, joints, scene_xyz, gazes, rag_metadata) if self.use_rag and rag_enabled else None
        # with torch.no_grad():
        ori, trans = self.traj_predictor(motions, joints, scene_xyz, gazes, occ)     # B,T,3  B,T,3
        
        # motions_cat = torch.cat([motions, joints.view(B, 6, -1)], dim=-1)   # B，6，69 + 38
        motion_extend = motions[:, [-1]].repeat(1, self.output_seq_len, 1)
        motions_cat = torch.cat([motions, motion_extend], dim=1)        #B，16，69 + 38

        human_points = self.extractor.cut_by_step(scene_xyz, trans, self.out_unit, ori)      # B,10 // unit,N,3
        scene_feats_batch = self._extract_scene_features(human_points.view(B * self.output_seq_len // self.out_unit, -1, 3))        # [B, num_points, feat_dim]
        scene_feats_batch = scene_feats_batch.view(B, self.output_seq_len // self.out_unit, -1, self.config.scene_feats_dim)
        # scene_feats = self._extract_scene_features(scene_xyz) 
        
        latent_full = motions_cat.clone()
        future_hidden = []
        for step in range(self.output_seq_len // self.out_unit):
            scene_feats = scene_feats_batch[:, step]
            
            motions_cat, start, end = self._update_motion_sequence(latent_full, step)
            motions = self._encode_motion_sequence(motions_cat)  #B,8,dim

            #scene_xyz_normalized = human_centered_scene(human_points[:, step],ori[:, start:end],trans[:,start:end])          # bs * len * N * 3
            spatial_prior = self._compute_spatial_prior(human_points[:, step])
            out_sca = self._apply_sca_blocks(motions, scene_feats, spatial_prior)
            if rag is not None and (not self.pose_bank or self.endpoint_adapter):
                guided, adapter_gate = self.prototype_adapter(
                    out_sca[:, -self.out_unit:], rag['prototype_tokens'],
                    rag['candidate_valid_mask'].to(out_sca.device), start=step * self.out_unit,
                )
                out_sca = torch.cat([out_sca[:, :-self.out_unit], guided], dim=1)
            if return_aux:
                future_hidden.append(out_sca[:, -self.out_unit:])
            pose = self.pose_fc(out_sca)
            
            latent_pred = torch.cat([ori[:, start:end], trans[:, start:end], pose], dim=-1) 
            # Keep the full sequence length fixed; only replace the current future block.
            latent_full = torch.cat(
                [latent_full[:, :end - self.out_unit], latent_pred[:, -self.out_unit :], latent_full[:, end:]],
                dim=1,
            )
          
        recons_joints = latent_to_joints(latent_full, self.vposer, self.smplx_model)[:, :, :23]
        recons_joints = torch.cat([joints, recons_joints[:, 6:]], dim=1).clone()
        pred_joints = recons_joints[:, :, :23]
        
        if self.config.traj_fusion:    
            pred_joints = self.pose_traj_fusion(trans.unsqueeze(2), pred_joints)
            
        pred_joints = self.motion_decoder(pred_joints)
        pre_refiner_joints = pred_joints
        refiner_gate = pred_joints.new_zeros(B)
        adapter_gate = pred_joints.new_zeros(B)
        refiner_delta = None
        if self.use_refiner and rag_enabled and self.motion_refiner is not None and self.pose_bank:
            refined_future, refiner_delta = self.motion_refiner(
                pred_joints[:, self.input_seq_len:], rag['endpoint_poses'].to(pred_joints),
                rag['candidate_valid_mask'].to(pred_joints.device), joints,
                **({'gazes': gazes, 'query_context': rag['query'].to(pred_joints)}
                   if self.config.rag_refiner_type in ('pose_endpoint_terminal', 'pose_endpoint_local') else {}))
            refiner_gate = (self.motion_refiner.trust_logits.sigmoid().mean().expand(B)
                            if hasattr(self.motion_refiner, 'trust_logits') else pred_joints.new_zeros(B))
            pred_joints = torch.cat([pred_joints[:, :self.input_seq_len], refined_future], dim=1)
        elif self.use_refiner and rag_enabled and self.motion_refiner is not None:
            if rag is None:
                proto_tokens = None
                query_context = None
                scores = None
                valid_mask = None
            else:
                proto_tokens = rag['prototype_tokens'].to(pred_joints)
                query_context = rag['query'].to(pred_joints)
                scores = rag['retrieval_scores'].to(pred_joints)
                valid_mask = rag['candidate_valid_mask'].to(device=pred_joints.device, dtype=torch.bool)
                if self.config.rag_forecast_rerank:
                    scores = forecast_candidate_scores(pred_joints[:, self.input_seq_len:],
                                                       rag['world_future_joints'], valid_mask,
                                                       local=self.config.rag_refiner_type == 'local')
            endpoint_args = {}
            if self.config.rag_refiner_type == 'endpoint' and rag is not None:
                with torch.no_grad():
                    endpoint_future, endpoint_valid = endpoint_candidates(self.rag_bank, joints, rag_metadata)
                endpoint_args = {'endpoint_future': endpoint_future, 'endpoint_valid': endpoint_valid}
            refined_future, refiner_gate, refiner_delta = self.motion_refiner(
                history_motions, pred_joints[:, self.input_seq_len:], proto_tokens, query_context, scores, valid_mask,
                **({'history_joints': joints} if self.config.rag_refiner_type in ('local','adaptive_terminal') else {}),
                **({'prototype_future': rag['world_future_joints']} if self.config.rag_refiner_type in ('blend', 'endpoint', 'terminal_blend','adaptive_terminal') and rag is not None else {}),
                **endpoint_args,
            )
            pred_joints = torch.cat([pred_joints[:, :self.input_seq_len], refined_future], dim=1)
        if return_aux:
            aux = {
                'pre_refiner_joints': pre_refiner_joints,
                'future_hidden': torch.cat(future_hidden, dim=1),
                'refiner_delta': refiner_delta,
                'refiner_gate': refiner_gate,
                'adapter_gate': adapter_gate if rag is not None else pred_joints.new_zeros(B),
            }
            if rag is not None:
                aux.update({
                    'retrieved_ids': rag['retrieved_ids'],
                    'retrieval_scores': rag['retrieval_scores'],
                    'candidate_valid_mask': rag['candidate_valid_mask'],
                    'filter_audit': rag['filter_audit'],
                })
            return latent_full, pred_joints, aux
        return latent_full, pred_joints
        
    def pose_traj_fusion(self, traj, joints):
        predictions = joints.clone()
        distance = predictions[:,:,[0],:] - traj
        predictions =  predictions - distance
        predictions[:, :, [0], :] = traj
        return predictions

class TIABlock(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.motion_encoder = SelfAttentionLayer(num_heads=8, num_q_channels=feat_dim, dropout=0.0)
        self.ia_attention = IntentionAwareAttention(feat_dim, num_head)
        self.mlp = Mlp(feat_dim * 3, feat_dim, drop=0.0)

    def forward(self, motion, scene_xyz, scene_feats, gaze_embedding):
        motion_tia = self.motion_encoder(motion)
        sm_feature = self.ia_attention(motion_tia, scene_feats)
        out = torch.cat([motion_tia, sm_feature, gaze_embedding], dim=-1)
        out = self.mlp(out)
        return out


class IntentionAwareAttention(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.motion_q = nn.Linear(feat_dim, feat_dim)
        self.scene_kv = nn.Linear(feat_dim, feat_dim * 2)
        self.out_emb = nn.Linear(feat_dim, feat_dim)

    def forward(self, motion, scene_feats):
        bs, len, _ = motion.shape
        N = scene_feats.shape[1]

        q = self.motion_q(motion)        # bs * len * c
        q = q * (self.feat_dim ** -0.5)
        k, v = torch.split(self.scene_kv(scene_feats), [self.feat_dim, self.feat_dim], dim=-1)
        # bs * head * len * _c, bs * head * N * _c
        q, k, v = q.view(bs, self.num_head, len, -1), k.view(bs, N, self.num_head, -1).transpose(1, 2), v.view(bs, N, self.num_head, -1).transpose(1, 2)

        att = torch.matmul(q, k.transpose(2, 3))       # bs * head * len * N
        att = F.softmax(att, dim=-1)                                # bs * head * N
        out = torch.matmul(att, v).view(bs, len, -1)           # bs, len, c
        out = self.out_emb(out)

        return out


class SCABlock(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.motion_encoder = SelfAttentionLayer(num_heads=8, num_q_channels=feat_dim, dropout=0.0)
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
        q, k, v = q.view(bs, self.num_head, len, -1), k.view(bs, N, self.num_head, -1).transpose(1, 2), v.view(bs, N, self.num_head, -1).transpose(1, 2)

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
        out = F.relu(self.fc1(xyz))
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
