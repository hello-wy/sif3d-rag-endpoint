"""Build a train-only single-frame GIMO or GIMO+AMASS pose bank."""

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.endpoint_pose_bank import EndpointPoseBank, EndpointPoseItem
from model.multimodal_rag import reference_frame, bank_manifest_hash
from scripts.rag_data import load_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--include_amass', action='store_true')
    parser.add_argument('--amass_root', default='/data/xiawangjie/humanml/amass_data')
    parser.add_argument('--babel_labels', default='/data/wuyang/BABEL/babel_gimo_clean')
    parser.add_argument('--smplx_path', default='/home/customer/sif3d-raw/data')
    parser.add_argument('--vposer_path', default='/home/customer/sif3d-raw/vposer_v1_0')
    parser.add_argument('--max_amass_items', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = load_split(args.data, 'train')
    bank = EndpointPoseBank.from_gimo_data(data)
    if args.include_amass:
        import smplx
        from human_body_prior.tools.model_loader import load_vposer
        from scripts.build_amass_motion_bank import babel_paths, convert_window

        paths = babel_paths(Path(args.amass_root), Path(args.babel_labels))
        random.Random(args.seed).shuffle(paths)
        vposer, _ = load_vposer(args.vposer_path, vp_model='snapshot')
        vposer.eval()
        body = smplx.create(args.smplx_path, model_type='smplx', gender='neutral',
                            ext='npz', num_pca_comps=12, batch_size=16).eval()
        added = 0
        for path, duration in paths:
            if duration < 8:
                continue
            result = convert_window(path, vposer, body, torch.device('cpu'))
            if result is None:
                continue
            start, motion, joints = result
            _, basis = reference_frame(joints[:6])
            endpoint = joints[-1]
            name = str(path.relative_to(args.amass_root))
            bank.items.append(EndpointPoseItem(
                sample_id=f'AMASS:{name}:{start}', dataset_name='AMASS', split='train',
                sequence_id=f'AMASS:{name}', scene_id='AMASS',
                history_frames=tuple(range(start, start + 6)),
                future_frames=tuple(range(start + 6, start + 16)),
                pose_motion=motion[-1].clone(),
                pose_joints=((endpoint - endpoint[:1]) @ basis).clone()))
            added += 1
            if args.max_amass_items and added >= args.max_amass_items:
                break
        if not added:
            raise ValueError('No eligible AMASS train recordings found')
        bank.manifest.update(dataset_name='GIMO+AMASS', amass_items=added,
                             amass_split='BABEL train, excluding validation/test recordings')
    bank.manifest.update(gimo_items=len(data['metadata']), retriever_trained=False,
                         sample_ids_hash=bank_manifest_hash(bank.items))
    bank.save(args.output)
    print(f'Saved {len(bank)} terminal poses to {args.output}')


if __name__ == '__main__':
    main()
