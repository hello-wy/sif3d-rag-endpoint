from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import smplx
import torch
from human_body_prior.tools.model_loader import load_vposer
from torch.utils.data import DataLoader

from config.config import MotionFromGazeConfig
from dataset import gimo_dataset
from model.sif3d_mod import SIF3D


def parse_args():
    parser = MotionFromGazeConfig()
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to SIF3D checkpoint (.pt/.pth). Falls back to --load_model_dir.')
    parser.add_argument('--vis_output_dir', type=str, default='outputs/sca_attention', help='Directory for OBJ outputs.')
    parser.add_argument('--max_batches', type=int, default=1, help='Maximum number of eval batches to visualize.')
    parser.add_argument('--attention_reduce', choices=['last_query', 'mean_query'], default='last_query', help='How to reduce the query/time dimension.')
    parser.add_argument('--head_reduce', choices=['mean', 'max'], default='mean', help='How to reduce the attention head dimension.')
    parser.add_argument('--save_gradcam', action='store_true', help='Also save grad-weighted attention OBJ files.')
    parser.add_argument('--grad_target', choices=['last_frame_l2', 'last_root_l2', 'last_pose_l2'], default='last_frame_l2', help='Backward target used when --save_gradcam is enabled.')
    parser.add_argument('--point_size', type=float, default=10.0, help='Unused for OBJ export; kept for CLI compatibility.')
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


def build_backward_target(latent_full, pred_joints, mode):
    if mode == 'last_pose_l2':
        return latent_full[:, -1].pow(2).sum()
    if mode == 'last_root_l2':
        return pred_joints[:, -1, 0].pow(2).sum()
    if mode == 'last_frame_l2':
        return pred_joints[:, -1].pow(2).sum()
    raise ValueError(f'Unsupported grad target: {mode}')


def reduce_attention_map(attention_probs, attention_reduce='last_query', head_reduce='mean'):
    if attention_reduce == 'last_query':
        reduced = attention_probs[:, :, -1, :]
    elif attention_reduce == 'mean_query':
        reduced = attention_probs.mean(dim=2)
    else:
        raise ValueError(f'Unsupported attention_reduce: {attention_reduce}')

    if head_reduce == 'mean':
        reduced = reduced.mean(dim=1)
    elif head_reduce == 'max':
        reduced = reduced.max(dim=1).values
    else:
        raise ValueError(f'Unsupported head_reduce: {head_reduce}')
    return reduced


def reduce_gradcam_map(attention_probs, attention_grads, attention_reduce='last_query', head_reduce='mean'):
    if attention_grads is None:
        return None

    if attention_reduce == 'last_query':
        probs = attention_probs[:, :, -1, :]
        grads = attention_grads[:, :, -1, :]
    elif attention_reduce == 'mean_query':
        probs = attention_probs.mean(dim=2)
        grads = attention_grads.mean(dim=2)
    else:
        raise ValueError(f'Unsupported attention_reduce: {attention_reduce}')

    gradcam = probs * grads.clamp(min=0)
    if head_reduce == 'mean':
        gradcam = gradcam.mean(dim=1)
    elif head_reduce == 'max':
        gradcam = gradcam.max(dim=1).values
    else:
        raise ValueError(f'Unsupported head_reduce: {head_reduce}')
    return gradcam


def normalize_map(values):
    vmin = values.min()
    vmax = values.max()
    if np.isclose(vmax, vmin):
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def save_attention_obj(points, weights, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized = normalize_map(weights)
    colors = matplotlib.colormaps['turbo'](normalized)[:, :3]

    with output_path.open('w', encoding='utf-8') as obj_file:
        obj_file.write(f'# {title}\n')
        obj_file.write('# Per-vertex colors encoded as RGB in [0, 1].\n')
        for point, color in zip(points, colors):
            obj_file.write(
                f'v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} '
                f'{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n'
            )


class SCAAttentionVisualizer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(config.vis_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        self.model.set_sca_attention_saving(True)

    def _save_batch_step_maps(self, seq, scene, step_idx, scene_points, raw_map, gradcam_map=None):
        batch_size = scene_points.shape[0]
        for batch_idx in range(batch_size):
            scene_name = sanitize_name(scene[batch_idx])
            seq_name = sanitize_name(seq[batch_idx])
            sample_dir = self.output_dir / scene_name / seq_name

            points_np = scene_points[batch_idx].detach().cpu().numpy()
            raw_np = raw_map[batch_idx].detach().cpu().numpy()
            raw_title = f'{scene_name}/{seq_name} | batch {batch_idx} | step {step_idx} | raw attention'
            raw_path = sample_dir / f'batch_{batch_idx:02d}_step_{step_idx:02d}_raw.obj'
            save_attention_obj(points_np, raw_np, raw_path, raw_title)

            if gradcam_map is not None:
                grad_np = gradcam_map[batch_idx].detach().cpu().numpy()
                grad_title = f'{scene_name}/{seq_name} | batch {batch_idx} | step {step_idx} | gradcam'
                grad_path = sample_dir / f'batch_{batch_idx:02d}_step_{step_idx:02d}_gradcam.obj'
                save_attention_obj(points_np, grad_np, grad_path, grad_title)

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
            scene_points = scene_points.to(self.device).contiguous()
            occ = occ.to(self.device)

            self.model.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(self.config.save_gradcam):
                latent_full, pred_joints, attention_debug = self.model(
                    poses_input,
                    joints_input[:, :, :23],
                    scene_points,
                    gazes,
                    occ,
                    return_attention=True,
                )

                if self.config.save_gradcam:
                    target = build_backward_target(latent_full, pred_joints, self.config.grad_target)
                    target.backward()

            for step_info in attention_debug['steps']:
                attention_probs = step_info['attention_probs']
                raw_map = reduce_attention_map(
                    attention_probs,
                    attention_reduce=self.config.attention_reduce,
                    head_reduce=self.config.head_reduce,
                )
                gradcam_map = None
                if self.config.save_gradcam:
                    gradcam_map = reduce_gradcam_map(
                        attention_probs,
                        attention_probs.grad,
                        attention_reduce=self.config.attention_reduce,
                        head_reduce=self.config.head_reduce,
                    )
                self._save_batch_step_maps(
                    seq=seq,
                    scene=scene,
                    step_idx=step_info['step'],
                    scene_points=step_info['scene_points'],
                    raw_map=raw_map,
                    gradcam_map=gradcam_map,
                )

            processed += 1
            print(f'Processed batch {batch_idx}, saved attention OBJ files to {self.output_dir}')


def main():
    config = parse_args()
    visualizer = SCAAttentionVisualizer(config)
    visualizer.run()


if __name__ == '__main__':
    main()
