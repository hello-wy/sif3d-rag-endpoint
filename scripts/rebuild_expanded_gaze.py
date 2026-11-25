"""Repair time-expanded GIMO gaze caches from face-free point-cloud PLYs."""

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import shutil

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.expand_endpoint_training_data import read_gaze_point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True,
                        help='Original GIMO cache for exact-window gaze cross-checks')
    parser.add_argument('--raw', type=Path, default=Path('/data/wuyang/human/GIMO'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = pd.read_csv(args.source / 'train.csv')
    original = pd.read_csv(args.reference / 'train.csv')
    original_gaze = torch.load(args.reference / 'cache/train/gazes.pth', map_location='cpu')
    original_by_id = {f'{row.sequence_path}:{int(row.start_frame)}': n
                      for n, row in original.iterrows()}
    if len(original_gaze) != len(original):
        raise ValueError('Reference gaze cache is not row aligned')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    recording_info = {}
    point_cache = {}
    gazes = []
    validity = []
    counts = Counter()
    max_reference_difference = 0.
    for n, row in rows.iterrows():
        recording = (str(row.scene), str(row.sequence_path), str(row.transformation))
        if recording not in recording_info:
            scene, sequence, transformation = recording
            directory = args.raw / scene / sequence
            scale = json.loads((directory / transformation).read_text())['scale']
            norm = np.loadtxt(args.raw / scene / 'scene_obj/transform_norm.txt').reshape(4, 4)
            norm[:3, 3] /= scale
            recording_info[recording] = directory, scale, norm
        directory, scale, norm = recording_info[recording]
        points = []
        for frame in (int(row.start_frame) + 15 * j for j in range(6)):
            key = (*recording, frame)
            if key not in point_cache:
                point_cache[key] = read_gaze_point(directory, frame, scale, norm)
            points.append(point_cache[key])
        valid = torch.tensor([point is not None for point in points], dtype=torch.bool)
        if not valid.any():
            raise ValueError(f'No valid gaze frame for {row.sequence_path}:{row.start_frame}')
        mean = torch.stack([point for point in points if point is not None]).mean(0)
        gaze = torch.stack([point if point is not None else mean for point in points])[:, None]
        sample_id = f'{row.sequence_path}:{int(row.start_frame)}'
        if sample_id in original_by_id:
            reference = original_gaze[original_by_id[sample_id]]
            difference = float((gaze - reference).abs().max())
            max_reference_difference = max(max_reference_difference, difference)
            if difference > 1e-4:
                raise ValueError(f'Original gaze mismatch for {sample_id}: {difference}')
        gazes.append(gaze)
        validity.append(valid)
        counts[int(valid.sum())] += 1
        if n % 200 == 0:
            print(f'processed {n + 1}/{len(rows)} windows', flush=True)
    repaired = torch.stack(gazes)
    if repaired.shape != (len(rows), 6, 1, 3):
        raise ValueError('Rebuilt gaze tensor shape mismatch')
    for split in ('train', 'valid', 'test'):
        shutil.copy2(args.source / f'{split}.csv', args.output / f'{split}.csv')
        source_cache = args.source / 'cache' / split
        target_cache = args.output / 'cache' / split
        target_cache.mkdir(parents=True)
        for file in source_cache.iterdir():
            if file.name in ('gazes.pth', 'gaze_valid.pth') and split == 'train':
                continue
            shutil.copy2(file, target_cache / file.name)
    torch.save(repaired, args.output / 'cache/train/gazes.pth')
    torch.save(torch.stack(validity), args.output / 'cache/train/gaze_valid.pth')
    displacement = (repaired[:, -1, 0] - repaired[:, 0, 0]).norm(dim=-1)
    report = {'source': str(args.source), 'reference': str(args.reference),
              'raw': str(args.raw), 'samples': len(rows),
              'valid_frame_counts': dict(sorted(counts.items())),
              'max_original_gaze_difference_m': max_reference_difference,
              'mean_observed_gaze_displacement_m': float(displacement.mean()),
              'median_observed_gaze_displacement_m': float(displacement.median())}
    (args.output / 'gaze_repair.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
