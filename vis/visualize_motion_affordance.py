from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch
from human_body_prior.tools.model_loader import load_vposer
from torch.utils.data import DataLoader

from config.config import MotionFromGazeConfig
from dataset import gimo_dataset
from model.sif3d_mod import SIF3D
from utils.motion_affordance import (
    MotionAffordanceConfig,
    SMPLX_BODY_EDGES,
    compute_motion_affordance,
    save_affordance_npz,
    save_colored_point_cloud_obj,
)
from utils.vis_utils import visualize_SMPLXjoints


def parse_args():
    parser = MotionFromGazeConfig()
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to SIF3D checkpoint (.pt/.pth). Falls back to --load_model_dir.')
    parser.add_argument('--vis_output_dir', type=str, default='outputs/motion_affordance', help='Directory for affordance outputs.')
    parser.add_argument('--max_batches', type=int, default=1, help='Maximum number of eval batches to visualize.')
    parser.add_argument('--sequence_source', choices=['pred_future', 'pred_full', 'gt_future', 'gt_full'], default='pred_future', help='Which motion sequence should drive the affordance map.')
    parser.add_argument('--frame_stride', type=int, default=1, help='Temporal stride used when turning joints into soft emitters.')
    parser.add_argument('--bone_samples', type=int, default=3, help='Number of interpolated samples per bone segment.')
    parser.add_argument('--trail_samples', type=int, default=4, help='Number of interpolated samples between consecutive frames for trail joints.')
    parser.add_argument('--sigma_joint_scale', type=float, default=3.2, help='Joint kernel width as a multiple of scene point spacing.')
    parser.add_argument('--sigma_bone_scale', type=float, default=4.4, help='Bone kernel width as a multiple of scene point spacing.')
    parser.add_argument('--sigma_trail_scale', type=float, default=5.2, help='Trail kernel width as a multiple of scene point spacing.')
    parser.add_argument('--vertical_sigma_scale', type=float, default=1.8, help='Vertical blur scale relative to the horizontal kernel width.')
    parser.add_argument('--gamma', type=float, default=2.0, help='Controls how quickly the accumulated field saturates.')
    parser.add_argument('--bone_weight', type=float, default=0.65, help='Relative weight for bone emitters.')
    parser.add_argument('--trail_weight', type=float, default=0.45, help='Relative weight for temporal trail emitters.')
    parser.add_argument('--diffusion_k', type=int, default=16, help='Number of neighbors used for point-cloud diffusion smoothing.')
    parser.add_argument('--diffusion_eta', type=float, default=0.35, help='Blend factor for each diffusion iteration.')
    parser.add_argument('--diffusion_iters', type=int, default=4, help='Number of diffusion smoothing iterations.')
    parser.add_argument('--max_points_for_diffusion', type=int, default=8192, help='Skip explicit kNN diffusion above this many scene points.')
    parser.add_argument('--clip_percentile', type=float, default=97.0, help='Percentile used to normalize the affordance field.')
    parser.add_argument('--preview_point_size', type=float, default=14.0, help='Base point size used by the top-down PNG preview.')
    parser.add_argument('--no_preview', action='store_true', help='Skip saving the top-down PNG preview.')
    parser.add_argument('--save_motion_sequence', action='store_true', help='Also export the sampled motion sequence as a skeleton OBJ.')
    parser.add_argument('--save_npz', action='store_true', help='Save a portable NPZ bundle for Blender or downstream tools.')
    return parser.parse_args()


def sanitize_name(value):
    text = str(value)
    safe = []
    for char in text:
        if char.isalnum() or char in ('-', '_'):
            safe.append(char)
        else:
            safe.append('_')
    return ''.join(safe).strip('_') or 'sample'


def load_checkpoint(checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict):
        for key in ('state_dict', 'model', 'model_state_dict'):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError(f'Unsupported checkpoint format: {type(state_dict)}')

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        if 'smplx' in key:
            continue
        cleaned[key] = value
    return cleaned


def build_affordance_config(args):
    return MotionAffordanceConfig(
        frame_stride=args.frame_stride,
        bone_samples=args.bone_samples,
        trail_samples=args.trail_samples,
        sigma_joint_scale=args.sigma_joint_scale,
        sigma_bone_scale=args.sigma_bone_scale,
        sigma_trail_scale=args.sigma_trail_scale,
        vertical_sigma_scale=args.vertical_sigma_scale,
        gamma=args.gamma,
        bone_weight=args.bone_weight,
        trail_weight=args.trail_weight,
        diffusion_k=args.diffusion_k,
        diffusion_eta=args.diffusion_eta,
        diffusion_iters=args.diffusion_iters,
        max_points_for_diffusion=args.max_points_for_diffusion,
        clip_percentile=args.clip_percentile,
        metadata={'sequence_source': args.sequence_source},
    )


def select_motion_sequence(pred_joints, joints_input, joints_label, sequence_source, input_seq_len):
    pred_joints = pred_joints[:, :, :23]
    history = joints_input[:, :, :23]
    future = joints_label[:, :, :23]
    gt_full = torch.cat([history, future], dim=1)

    if sequence_source == 'pred_future':
        return pred_joints[:, input_seq_len:]
    if sequence_source == 'pred_full':
        return pred_joints
    if sequence_source == 'gt_future':
        return future
    if sequence_source == 'gt_full':
        return gt_full
    raise ValueError(f'Unsupported sequence_source: {sequence_source}')


def _draw_pose_bev(ax, pose_xy, alpha_scale):
    color = (0.20, 0.84, 0.34, alpha_scale)
    for start_idx, end_idx in SMPLX_BODY_EDGES:
        if start_idx >= pose_xy.shape[0] or end_idx >= pose_xy.shape[0]:
            continue
        ax.plot(
            [pose_xy[start_idx, 0], pose_xy[end_idx, 0]],
            [pose_xy[start_idx, 1], pose_xy[end_idx, 1]],
            color=color,
            linewidth=1.5,
        )
    ax.scatter(pose_xy[:, 0], pose_xy[:, 1], c=[color], s=12.0, linewidths=0)


def save_affordance_preview(points, scores, colors, motion_joints, output_path, title, point_size):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points)
    scores = np.asarray(scores)
    colors = np.asarray(colors)
    motion_joints = np.asarray(motion_joints)

    order = np.argsort(scores)
    points = points[order]
    scores = scores[order]
    colors = colors[order]

    glow_alpha = (0.010 + 0.055 * np.power(scores, 0.45))[:, None]
    haze_alpha = (0.025 + 0.120 * np.power(scores, 0.55))[:, None]
    core_alpha = (0.060 + 0.720 * np.power(scores, 0.78))[:, None]
    glow_rgba = np.concatenate([colors, glow_alpha], axis=1)
    haze_rgba = np.concatenate([colors, haze_alpha], axis=1)
    core_rgba = np.concatenate([colors, core_alpha], axis=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(points[:, 0], points[:, 2], c=[(0.82, 0.82, 0.82, 0.08)], s=max(point_size * 0.12, 1.0), linewidths=0)
    ax.scatter(points[:, 0], points[:, 2], c=glow_rgba, s=point_size * 11.0, linewidths=0)
    ax.scatter(points[:, 0], points[:, 2], c=haze_rgba, s=point_size * 4.4, linewidths=0)
    ax.scatter(points[:, 0], points[:, 2], c=core_rgba, s=point_size * 1.1, linewidths=0)

    if motion_joints.size > 0:
        frame_count = motion_joints.shape[0]
        frame_ids = np.unique(np.linspace(0, frame_count - 1, num=min(6, frame_count), dtype=int))
        for local_rank, frame_idx in enumerate(frame_ids, start=1):
            pose = motion_joints[frame_idx, :, [0, 2]]
            _draw_pose_bev(ax, pose, alpha_scale=0.06 + 0.04 * local_rank)

        pelvis_path = motion_joints[:, 0, :]
        ax.plot(pelvis_path[:, 0], pelvis_path[:, 2], color=(0.18, 0.76, 0.32, 0.38), linewidth=2.2)
        ax.scatter(pelvis_path[:, 0], pelvis_path[:, 2], c=[(0.18, 0.76, 0.32, 0.18)], s=18.0, linewidths=0)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


class MotionAffordanceVisualizer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(config.vis_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.affordance_config = build_affordance_config(config)

        self.test_loader = DataLoader(
            gimo_dataset.EgoEvalDataset(config, train=False),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        self.vposer, _ = load_vposer(config.vposer_path, vp_model='snapshot')
        self.vposer = self.vposer.to(self.device)
        self.body_model = smplx.create(
            config.smplx_path,
            model_type='smplx',
            gender='neutral',
            ext='npz',
            num_pca_comps=12,
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_expression=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_transl=True,
            num_betas=10,
            num_expression_coeffs=10,
            batch_size=config.batch_size * (config.input_seq_len + config.output_seq_len),
        ).to(self.device)

        self.model = SIF3D(config, self.vposer, self.body_model).to(self.device)
        checkpoint_path = config.checkpoint_path or config.load_model_dir
        if checkpoint_path is None:
            raise ValueError('Please provide --checkpoint_path or --load_model_dir.')
        state_dict = load_checkpoint(checkpoint_path, self.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f'Loaded checkpoint: {checkpoint_path}')
        if missing:
            print(f'Missing keys: {len(missing)}')
        if unexpected:
            print(f'Unexpected keys: {len(unexpected)}')
        self.model.eval()

    def _save_sample_outputs(self, scene_name, seq_name, batch_idx, scene_points, motion_joints, affordance):
        sample_dir = self.output_dir / scene_name / seq_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        title = f'{scene_name}/{seq_name} | batch {batch_idx} | {self.config.sequence_source}'

        obj_path = sample_dir / f'batch_{batch_idx:02d}_motion_affordance.obj'
        save_colored_point_cloud_obj(scene_points, affordance['colors'], obj_path, title)

        metadata_path = sample_dir / f'batch_{batch_idx:02d}_motion_affordance.json'
        with metadata_path.open('w', encoding='utf-8') as handle:
            json.dump(affordance['parameters'], handle, indent=2)

        if self.config.save_npz:
            npz_path = sample_dir / f'batch_{batch_idx:02d}_motion_affordance.npz'
            save_affordance_npz(
                npz_path,
                scene_points=scene_points,
                scores=affordance['scores'],
                colors=affordance['colors'],
                motion_joints=motion_joints,
                anchors=affordance['anchors'],
                anchor_weights=affordance['anchor_weights'],
                parameters=affordance['parameters'],
            )

        if not self.config.no_preview:
            preview_path = sample_dir / f'batch_{batch_idx:02d}_motion_affordance_preview.png'
            save_affordance_preview(
                scene_points,
                affordance['scores'],
                affordance['colors'],
                motion_joints,
                preview_path,
                title,
                point_size=self.config.preview_point_size,
            )

        if self.config.save_motion_sequence:
            motion_path_tag = f'batch_{batch_idx:02d}_motion_sequence'
            visualize_SMPLXjoints(np.asarray(motion_joints), None, motion_path_tag, output_path=str(sample_dir))

    def run(self):
        processed = 0
        for batch_idx, data in enumerate(self.test_loader):
            if processed >= self.config.max_batches:
                break

            gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene, occ = data
            if self.config.eval_scene != 'all':
                selected = [self.config.eval_scene in sample_scene for sample_scene in scene]
                if not any(selected):
                    continue

            gazes = gazes.to(self.device)
            poses_input = poses_input.to(self.device)
            joints_input = joints_input.to(self.device)
            joints_label = joints_label.to(self.device)
            scene_points = scene_points.to(self.device).contiguous()
            occ = occ.to(self.device)

            with torch.no_grad():
                _, pred_joints = self.model(
                    poses_input,
                    joints_input[:, :, :23],
                    scene_points,
                    gazes,
                    occ,
                )

            motion_sequence = select_motion_sequence(
                pred_joints=pred_joints,
                joints_input=joints_input,
                joints_label=joints_label,
                sequence_source=self.config.sequence_source,
                input_seq_len=self.config.input_seq_len,
            )

            for sample_idx in range(scene_points.shape[0]):
                scene_name = sanitize_name(scene[sample_idx])
                seq_name = sanitize_name(seq[sample_idx])
                points_np = scene_points[sample_idx].detach().cpu().numpy()
                motion_np = motion_sequence[sample_idx].detach().cpu().numpy()
                affordance = compute_motion_affordance(points_np, motion_np, self.affordance_config, return_details=True)
                affordance = {
                    key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                    for key, value in affordance.items()
                }
                self._save_sample_outputs(
                    scene_name=scene_name,
                    seq_name=seq_name,
                    batch_idx=sample_idx,
                    scene_points=points_np,
                    motion_joints=motion_np,
                    affordance=affordance,
                )

            processed += 1
            print(f'Processed batch {batch_idx}, saved motion affordance outputs to {self.output_dir}')


def main():
    config = parse_args()
    visualizer = MotionAffordanceVisualizer(config)
    visualizer.run()


if __name__ == '__main__':
    main()
