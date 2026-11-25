"""Single-frame pose retrieval for gaze and scene conditioned PMD refinement."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from model.multimodal_rag import MotionBank, reference_frame


VERSION = 'endpoint-pose-bank-v1'


@dataclass
class EndpointPoseItem:
    sample_id: str
    dataset_name: str
    split: str
    sequence_id: str
    scene_id: str
    history_frames: tuple
    future_frames: tuple
    pose_motion: torch.Tensor
    pose_joints: torch.Tensor

    def __post_init__(self):
        if self.split != 'train' or self.pose_motion.shape != (38,) or self.pose_joints.shape != (23, 3):
            raise ValueError('Endpoint bank requires train-only [38] motion and [23,3] joints')
        if not torch.isfinite(self.pose_motion).all() or not torch.isfinite(self.pose_joints).all():
            raise ValueError('Endpoint pose contains non-finite values')

    def metadata(self):
        return {key: getattr(self, key) for key in ('sample_id', 'dataset_name', 'split', 'sequence_id', 'scene_id', 'history_frames', 'future_frames')}


class EndpointPoseBank(MotionBank):
    """Only canonical terminal poses and metadata are persisted; keys are pose-encoded."""

    def __init__(self, items, keys=None, manifest=None, encoder_state_dict=None, pose_encoder_state_dict=None):
        super().__init__(items, keys, manifest, encoder_state_dict)
        if keys is not None and encoder_state_dict is None:
            raise ValueError('Trained endpoint keys require a query encoder checkpoint')
        if keys is not None and self.manifest.get('key_source', 'terminal_pose') == 'terminal_pose' and pose_encoder_state_dict is None:
            raise ValueError('Terminal-pose keys require a pose encoder checkpoint')
        self.manifest['bank_version'] = VERSION
        self.pose_encoder_state_dict = pose_encoder_state_dict

    @classmethod
    def from_gimo_data(cls, data, fps=2):
        items = []
        for n, row in data['rows'].iterrows():
            history = data['joints_input'][n]
            _, basis = reference_frame(history)
            endpoint = data['joints_label'][n, -1, :23]
            start = int(row.start_frame)
            items.append(EndpointPoseItem(
                sample_id=f'{row.sequence_path}:{start}', dataset_name='GIMO', split='train',
                sequence_id=str(row.sequence_path), scene_id=str(row.scene),
                history_frames=tuple(start + int(t * 30 / fps) for t in range(6)),
                future_frames=tuple(start + int((6 + t) * 30 / fps) for t in range(10)),
                pose_motion=data['poses_label'][n, -1].cpu().clone(),
                pose_joints=((endpoint - endpoint[:1]) @ basis).cpu().clone()))
        return cls(items, manifest={'dataset_name': 'GIMO', 'split': 'train', 'input_seq_len': 6,
                                     'output_seq_len': 10, 'fps': fps, 'sample_points': data['scenes'].shape[1],
                                     'use_motion': False, 'use_gaze': True, 'use_scene': True,
                                     'pose_space': 'last-observed-heading; pelvis-relative'})

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'version': VERSION, 'items': [dict(item.__dict__) for item in self.items],
                    'keys': self.keys, 'manifest': self.manifest,
                    'encoder_state_dict': self.encoder_state_dict,
                    'pose_encoder_state_dict': self.pose_encoder_state_dict}, path)

    def encode_keys(self, *args, **kwargs):
        raise ValueError('Endpoint pose keys must be built with EndpointPoseEncoder')

    def gather_future(self, *args, **kwargs):
        raise ValueError('Endpoint pose banks have no future sequence; use gather_pose')

    @classmethod
    def load(cls, path, map_location='cpu'):
        payload = torch.load(path, map_location=map_location)
        if payload.get('version') != VERSION:
            raise ValueError('Expected endpoint pose bank')
        return cls([EndpointPoseItem(**raw) for raw in payload['items']], payload.get('keys'),
                   payload.get('manifest'), payload.get('encoder_state_dict'), payload.get('pose_encoder_state_dict'))

    def gather_pose(self, ids, query_history):
        if ids.ndim != 2 or len(ids) != len(query_history):
            raise ValueError('IDs and query history batch mismatch')
        out = query_history.new_zeros(*ids.shape, 23, 3)
        for b, history in enumerate(query_history):
            origin, basis = reference_frame(history)
            for k, index in enumerate(ids[b].tolist()):
                if index >= 0:
                    out[b, k] = self.items[index].pose_joints.to(out) @ basis.T + origin
        return out


class EndpointPoseEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(69, 256), nn.GELU(), nn.LayerNorm(256), nn.Linear(256, embedding_dim))

    def forward(self, pose):
        return F.normalize(self.net(pose.flatten(1)), dim=-1)


class EndpointPoseRefiner(nn.Module):
    """Guide PMD's body pose toward retrieved terminal poses without moving its pelvis."""

    def __init__(self, frames=10, joints=23, rerank_topk=0, align_yaw=False):
        super().__init__()
        self.frames = frames
        self.rerank_topk = rerank_topk
        self.align_yaw = align_yaw
        self.trust_logits = nn.Parameter(torch.full((frames, joints, 1), -2.0))
        self.residual = nn.Sequential(nn.Linear(joints * 12 + 1, 128), nn.GELU(), nn.Dropout(.2),
                                      nn.Linear(128, joints * 3 + 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, base_future, candidate_poses, candidate_valid_mask, history_joints=None):
        if base_future.ndim != 4 or candidate_poses.shape[-2:] != (base_future.shape[2], 3):
            raise ValueError('Expected base [B,T,J,3] and poses [B,K,J,3]')
        valid = candidate_valid_mask.bool()
        if self.rerank_topk and candidate_poses.shape[1] > self.rerank_topk:
            forecast_pose = base_future[:, -1] - base_future[:, -1, :1]
            candidates = candidate_poses - candidate_poses[:, :, :1]
            distance = (candidates - forecast_pose[:, None]).norm(dim=-1).mean(-1)
            chosen = distance.masked_fill(~valid, torch.inf).topk(self.rerank_topk, largest=False).indices
            candidate_poses = candidate_poses.gather(1, chosen[:, :, None, None].expand(-1, -1, candidates.shape[2], 3))
            valid = valid.gather(1, chosen)
        if self.align_yaw:
            forecast_pose = base_future[:, -1] - base_future[:, -1, :1]
            candidate_relative = candidate_poses - candidate_poses[:, :, :1]
            horizontal_dot = (candidate_relative[..., 0] * forecast_pose[:, None, :, 0] +
                              candidate_relative[..., 2] * forecast_pose[:, None, :, 2]).sum(-1)
            horizontal_cross = (candidate_relative[..., 2] * forecast_pose[:, None, :, 0] -
                                candidate_relative[..., 0] * forecast_pose[:, None, :, 2]).sum(-1)
            angle = torch.atan2(horizontal_cross, horizontal_dot)
            cosine, sine = angle.cos()[..., None], angle.sin()[..., None]
            candidate_poses = torch.stack((cosine * candidate_relative[..., 0] + sine * candidate_relative[..., 2],
                                           candidate_relative[..., 1],
                                           -sine * candidate_relative[..., 0] + cosine * candidate_relative[..., 2]), -1)
        weights = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
        pose = ((candidate_poses - candidate_poses[:, :, :1]) * weights[:, :, None, None]).sum(1)
        relative = base_future - base_future[:, :, :1]
        terminal = relative[:, -1]
        observed = (history_joints[:, -1] - history_joints[:, -1, :1]
                    if history_joints is not None else torch.zeros_like(terminal))
        time = torch.linspace(0, 1, base_future.shape[1], device=base_future.device,
                              dtype=base_future.dtype)[None, :, None].expand(len(base_future), -1, -1)
        inputs = torch.cat([relative.flatten(2), terminal.flatten(1)[:, None].expand(-1, len(time[0]), -1),
                            pose.flatten(1)[:, None].expand(-1, len(time[0]), -1),
                            observed.flatten(1)[:, None].expand(-1, len(time[0]), -1), time], -1)
        raw = self.residual(inputs)
        angle = .5 * torch.tanh(raw[..., -1:])
        cosine, sine = angle.cos(), angle.sin()
        rotated = torch.stack((cosine * relative[..., 0] + sine * relative[..., 2],
                               relative[..., 1],
                               -sine * relative[..., 0] + cosine * relative[..., 2]), -1)
        residual = .12 * torch.tanh(raw[..., :-1].view_as(base_future))
        residual = residual - residual[:, :, :1]
        # The pose term adjusts PMD's terminal estimate while retaining its temporal motion.
        trust = self.trust_logits[:base_future.shape[1]].sigmoid()[None]
        delta = (rotated - relative) + trust * (pose[:, None] - rotated[:, -1:]) + residual
        delta = delta - delta[:, :, :1]
        delta = delta.masked_fill(~valid.any(-1)[:, None, None, None], 0)
        return base_future + delta, delta


class EndpointPoseTemporalRefiner(EndpointPoseRefiner):
    """Use the complete PMD forecast when distributing endpoint-pose corrections."""

    def __init__(self, frames=10, joints=23, rerank_topk=0, align_yaw=False):
        super().__init__(frames, joints, rerank_topk, align_yaw)
        self.residual = nn.Sequential(
            nn.Linear(joints * 12 + 1, 128), nn.GELU(),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 256, .1, batch_first=True), 1),
            nn.Linear(128, joints * 3 + 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)


class EndpointTrustRefiner(nn.Module):
    """Predict how much an endpoint pose should correct each PMD forecast."""

    def __init__(self, frames=10):
        super().__init__()
        self.frames = frames
        self.trust = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Dropout(.1), nn.Linear(32, 1))
        nn.init.zeros_(self.trust[-1].weight)
        self.trust[-1].bias.data.fill_(float(torch.logit(torch.tensor(.25 / 1.2))))
        self.register_buffer('feature_mean', torch.zeros(8))
        self.register_buffer('feature_std', torch.ones(8))

    def observation_features(self, base_future, candidate_poses, candidate_valid_mask, history_joints):
        if history_joints is None:
            raise ValueError('Endpoint trust requires observed joints for confidence estimation')
        valid = candidate_valid_mask.bool()
        weights = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
        candidates = candidate_poses - candidate_poses[:, :, :1]
        prior = (candidates * weights[:, :, None, None]).sum(1)
        relative = base_future - base_future[:, :, :1]
        endpoint = relative[:, -1]
        observed = history_joints[:, -1] - history_joints[:, -1, :1]
        gap = (prior - endpoint).norm(dim=-1)
        dispersion = ((candidates - prior[:, None]).norm(dim=-1).mean(-1) * weights).sum(-1)
        history_gap = (endpoint - observed).norm(dim=-1)
        prior_history_gap = (prior - observed).norm(dim=-1)
        velocity = (relative[:, -1] - relative[:, -2]).norm(dim=-1)
        features = torch.stack((gap.mean(-1), gap.std(-1), gap.max(-1).values,
                                dispersion, history_gap.mean(-1), prior_history_gap.mean(-1),
                                velocity.mean(-1), valid.float().mean(-1)), -1)
        return features, prior, relative

    def forward(self, base_future, candidate_poses, candidate_valid_mask, history_joints=None):
        features, prior, relative = self.observation_features(base_future, candidate_poses,
                                                              candidate_valid_mask, history_joints)
        trust = 1.2 * self.trust((features - self.feature_mean) / self.feature_std.clamp_min(.05)).sigmoid()
        ramp = torch.linspace(.25, 1., relative.shape[1], device=relative.device, dtype=relative.dtype)
        correction = (trust[:, None, None] * ramp[None, :, None, None]) * (prior[:, None] - relative[:, -1:])
        correction = correction - correction[:, :, :1]
        correction = correction.masked_fill(~candidate_valid_mask.bool().any(-1)[:, None, None, None], 0.)
        return base_future + correction, correction


class EndpointJointTrustRefiner(nn.Module):
    """Estimate retrieval reliability separately for each joint."""

    def __init__(self, frames=10, joints=23):
        super().__init__()
        self.frames = frames
        self.joints = joints
        self.trust = nn.Sequential(nn.Linear(20, 32), nn.GELU(), nn.Dropout(.1), nn.Linear(32, 1))
        nn.init.zeros_(self.trust[-1].weight)
        nn.init.zeros_(self.trust[-1].bias)
        self.joint_bias = nn.Parameter(torch.full((joints, 1), float(torch.logit(torch.tensor(.25 / 1.2)))))
        self.register_buffer('feature_mean', torch.zeros(20))
        self.register_buffer('feature_std', torch.ones(20))

    def observation_features(self, base_future, candidate_poses, candidate_valid_mask, history_joints):
        if history_joints is None:
            raise ValueError('Joint trust requires observed joints')
        valid = candidate_valid_mask.bool()
        weights = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
        candidates = candidate_poses - candidate_poses[:, :, :1]
        prior = (candidates * weights[:, :, None, None]).sum(1)
        relative = base_future - base_future[:, :, :1]
        endpoint = relative[:, -1]
        observed = history_joints[:, -1] - history_joints[:, -1, :1]
        dispersion = ((candidates - prior[:, None]).norm(dim=-1) * weights[:, :, None]).sum(1)
        gap = prior - endpoint
        history_gap = endpoint - observed
        velocity = relative[:, -1] - relative[:, -2]
        global_gap = gap.norm(dim=-1).mean(-1)[:, None, None].expand(-1, self.joints, 1)
        features = torch.cat((endpoint, prior, observed, gap, history_gap, velocity,
                              dispersion[..., None], global_gap), -1)
        return features, prior, relative

    def forward(self, base_future, candidate_poses, candidate_valid_mask, history_joints=None):
        features, prior, relative = self.observation_features(base_future, candidate_poses,
                                                              candidate_valid_mask, history_joints)
        trust = 1.2 * (self.trust((features - self.feature_mean) / self.feature_std.clamp_min(.05)) +
                       self.joint_bias[None]).sigmoid()
        ramp = torch.linspace(.25, 1., relative.shape[1], device=relative.device, dtype=relative.dtype)
        correction = trust[:, None] * ramp[None, :, None, None] * (prior[:, None] - relative[:, -1:])
        correction = correction - correction[:, :, :1]
        correction = correction.masked_fill(~candidate_valid_mask.bool().any(-1)[:, None, None, None], 0.)
        return base_future + correction, correction


class EndpointInterpolationRefiner(nn.Module):
    """Blend PMD with an observed-to-retrieved endpoint pose transition."""

    def __init__(self, frames=10):
        super().__init__()
        self.frames = frames
        self.mix_logit = nn.Parameter(torch.logit(torch.tensor(.3)))

    def forward(self, base_future, candidate_poses, candidate_valid_mask, history_joints=None):
        if history_joints is None:
            raise ValueError('Endpoint interpolation requires observed joints')
        valid = candidate_valid_mask.bool()
        weights = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
        candidates = candidate_poses - candidate_poses[:, :, :1]
        prior = (candidates * weights[:, :, None, None]).sum(1)
        observed = history_joints[:, -1] - history_joints[:, -1, :1]
        relative = base_future - base_future[:, :, :1]
        ramp = torch.linspace(.1, 1., base_future.shape[1], device=base_future.device, dtype=base_future.dtype)
        transition = (1 - ramp)[None, :, None, None] * observed[:, None] + ramp[None, :, None, None] * prior[:, None]
        correction = self.mix_logit.sigmoid() * (transition - relative)
        correction = correction - correction[:, :, :1]
        correction = correction.masked_fill(~valid.any(-1)[:, None, None, None], 0.)
        return base_future + correction, correction


class EndpointTerminalRefiner(EndpointInterpolationRefiner):
    """Correct the final pose after interpolation using gaze and retrieved context."""

    def __init__(self, frames=10, context_dim=256):
        super().__init__(frames)
        self.terminal = nn.Sequential(nn.Linear(23 * 3 * 3 + 3 + context_dim, 128),
                                      nn.GELU(), nn.Dropout(.2), nn.Linear(128, 70))
        nn.init.zeros_(self.terminal[-1].weight)
        nn.init.zeros_(self.terminal[-1].bias)

    def forward(self, base_future, candidate_poses, candidate_valid_mask, history_joints=None,
                gazes=None, query_context=None):
        if history_joints is None or gazes is None or query_context is None:
            raise ValueError('Terminal refiner requires observed joints, gaze, and query context')
        interpolated, initial_delta = super().forward(base_future, candidate_poses,
                                                       candidate_valid_mask, history_joints)
        valid = candidate_valid_mask.bool()
        weights = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
        candidate_relative = candidate_poses - candidate_poses[:, :, :1]
        prior = (candidate_relative * weights[:, :, None, None]).sum(1)
        observed = history_joints[:, -1] - history_joints[:, -1, :1]
        relative = interpolated - interpolated[:, :, :1]
        width = history_joints[:, -1, 2] - history_joints[:, -1, 1]
        horizontal = torch.stack((width[:, 0], torch.zeros_like(width[:, 0]), width[:, 2]), -1)
        fallback = horizontal.new_tensor([1., 0., 0.]).expand_as(horizontal)
        right = F.normalize(torch.where(horizontal.norm(dim=-1, keepdim=True) > 1e-5,
                                        horizontal, fallback), dim=-1)
        up = horizontal.new_tensor([0., 1., 0.]).expand_as(right)
        forward = torch.linalg.cross(right, up, dim=-1)
        basis = torch.stack((right, up, forward), -1)
        gaze = gazes[:, -1, 0] if gazes.ndim == 4 else gazes[:, -1]
        gaze_direction = torch.einsum('bc,bcd->bd', gaze - base_future[:, -1, 0], basis)
        features = torch.cat(((relative[:, -1] @ basis).flatten(1),
                              (prior @ basis).flatten(1), (observed @ basis).flatten(1),
                              gaze_direction, query_context), -1)
        raw = self.terminal(features)
        yaw = torch.tanh(raw[:, :1])
        pose_residual = .15 * torch.tanh(raw[:, 1:].reshape(-1, 23, 3))
        frames = min(4, relative.shape[1])
        ramp = relative.new_zeros(relative.shape[1])
        ramp[-frames:] = torch.linspace(1 / frames, 1, frames, device=relative.device, dtype=relative.dtype)
        angle = yaw[:, None, :] * ramp[None, :, None]
        cosine, sine = angle.cos(), angle.sin()
        local = relative @ basis[:, None]
        rotated = torch.stack((cosine * local[..., 0] + sine * local[..., 2],
                               local[..., 1],
                               -sine * local[..., 0] + cosine * local[..., 2]), -1)
        local_delta = rotated - local + ramp[None, :, None, None] * pose_residual[:, None]
        local_delta = local_delta - local_delta[:, :, :1]
        correction = local_delta @ basis.transpose(-1, -2)[:, None]
        correction = correction.masked_fill(~valid.any(-1)[:, None, None, None], 0.)
        return interpolated + correction, initial_delta + correction


class EndpointLocalMotionRefiner(EndpointInterpolationRefiner):
    """Learn a short pose correction sequence in the observed body frame."""

    def __init__(self, frames=10):
        super().__init__(frames)
        self.input = nn.Linear(23 * 3 * 4 + 4, 128)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 4, 256, .2, batch_first=True), 1)
        self.output = nn.Linear(128, 70)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, base_future, candidate_poses, candidate_valid_mask, history_joints=None,
                gazes=None, query_context=None):
        if history_joints is None or gazes is None:
            raise ValueError('Local endpoint refiner requires observed joints and gaze')
        interpolated, initial_delta = super().forward(base_future, candidate_poses,
                                                       candidate_valid_mask, history_joints)
        valid = candidate_valid_mask.bool()
        weights = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
        candidates = candidate_poses - candidate_poses[:, :, :1]
        prior = (candidates * weights[:, :, None, None]).sum(1)
        observed = history_joints[:, -1] - history_joints[:, -1, :1]
        width = history_joints[:, -1, 2] - history_joints[:, -1, 1]
        horizontal = torch.stack((width[:, 0], torch.zeros_like(width[:, 0]), width[:, 2]), -1)
        fallback = horizontal.new_tensor([1., 0., 0.]).expand_as(horizontal)
        right = F.normalize(torch.where(horizontal.norm(dim=-1, keepdim=True) > 1e-5,
                                        horizontal, fallback), dim=-1)
        up = horizontal.new_tensor([0., 1., 0.]).expand_as(right)
        basis = torch.stack((right, up, torch.linalg.cross(right, up, dim=-1)), -1)
        base_local = ((base_future - base_future[:, :, :1]) @ basis[:, None]).flatten(2)
        interp_local = ((interpolated - interpolated[:, :, :1]) @ basis[:, None]).flatten(2)
        prior_local = (prior @ basis).flatten(1)[:, None].expand(-1, base_future.shape[1], -1)
        observed_local = (observed @ basis).flatten(1)[:, None].expand_as(prior_local)
        gaze = gazes[:, -1, 0] if gazes.ndim == 4 else gazes[:, -1]
        gaze_local = torch.einsum('bc,bcd->bd', gaze - base_future[:, -1, 0], basis)
        gaze_local = gaze_local[:, None].expand(-1, base_future.shape[1], -1)
        time = torch.linspace(0, 1, base_future.shape[1], device=base_future.device,
                              dtype=base_future.dtype)[None, :, None].expand(len(base_future), -1, -1)
        inputs = torch.cat((base_local, interp_local, prior_local, observed_local, gaze_local, time), -1)
        raw = self.output(self.temporal(self.input(inputs)))
        angle = torch.tanh(raw[..., -1:])
        cosine, sine = angle.cos(), angle.sin()
        local_pose = interp_local.reshape_as(base_future)
        rotated = torch.stack((cosine * local_pose[..., 0] + sine * local_pose[..., 2],
                               local_pose[..., 1],
                               -sine * local_pose[..., 0] + cosine * local_pose[..., 2]), -1)
        local_delta = rotated - local_pose + .12 * torch.tanh(raw[..., :-1].view_as(base_future))
        local_delta = local_delta - local_delta[:, :, :1]
        correction = local_delta @ basis.transpose(-1, -2)[:, None]
        correction = correction.masked_fill(~valid.any(-1)[:, None, None, None], 0.)
        return interpolated + correction, initial_delta + correction

