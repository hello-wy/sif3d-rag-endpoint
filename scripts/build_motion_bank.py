"""Build a deterministic GIMO train-only MotionBank and frozen key index.

Example:
  python scripts/build_motion_bank.py --output runs/gimo_motion_bank.pt \
      --folder 318_4096 --traj_ckpts /path/to/traj.pth
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import MotionFromGazeConfig
from dataset.gimo_dataset import EgoEvalDataset
from model.multimodal_rag import MotionBank, MultiModalRetriever, bank_manifest_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    args, unknown = parser.parse_known_args()
    config = MotionFromGazeConfig().parse_args(unknown)
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    # Dataset construction reads cached tensors and never applies its online
    # random augmentation here.
    dataset = EgoEvalDataset(config, train=True)
    bank = MotionBank.from_gimo_dataset(dataset)
    modalities = str(config.rag_modalities)
    use_motion = 'motion' in modalities
    use_gaze = 'gaze' in modalities
    use_scene = 'scene' in modalities
    if config.rag_motion_only:
        use_motion, use_gaze, use_scene = True, False, False
    elif config.rag_motion_scene:
        use_motion, use_gaze, use_scene = True, False, True
    elif config.rag_motion_gaze:
        use_motion, use_gaze, use_scene = True, True, False
    retriever = MultiModalRetriever(
        motion_dim=38,
        joint_count=23,
        embedding_dim=args.embedding_dim,
        use_motion=use_motion,
        use_gaze=use_gaze,
        use_scene=use_scene,
        dropout=0.0,
    )
    bank.encode_keys(retriever, batch_size=args.batch_size)
    bank.manifest.update({
        'dataset_name': 'GIMO',
        'split': 'train',
        'preprocess_version': 'gimo-cached-v1',
        'input_seq_len': config.input_seq_len,
        'output_seq_len': config.output_seq_len,
        'fps': config.fps,
        'sample_points': config.sample_points,
        'retriever_trained': False,
        'sample_ids_hash': bank_manifest_hash(bank.items),
        'random_augmentation': False,
        'scene_noise': False,
        'normalization': 'last-history-pelvis-and-heading; no learned cross-split statistics',
        'use_motion': use_motion,
        'use_gaze': use_gaze,
        'use_scene': use_scene,
    })
    bank.save(args.output)
    print(f'saved {len(bank)} train windows to {args.output}; key_hash={bank.encoder_hash()}')


if __name__ == '__main__':
    main()
