"""Train gaze+scene to endpoint-pose retrieval, then a frozen-PMD pose refiner."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.endpoint_pose_bank import EndpointPoseBank, EndpointPoseEncoder, EndpointPoseRefiner, EndpointPoseTemporalRefiner, EndpointTrustRefiner, EndpointJointTrustRefiner, EndpointInterpolationRefiner, EndpointTerminalRefiner, EndpointLocalMotionRefiner
from model.multimodal_rag import MultiModalRetriever
from scripts.rag_data import load_split, errors, METRICS


def query_features(encoder, data, device):
    parts = []
    with torch.no_grad():
        for start in range(0, len(data['metadata']), 8):
            end = start + 8
            features = encoder.prepare_features(None, None,
                                                data['gazes'][start:end], data['scenes'][start:end])
            parts.append(tuple(feature.cpu() for feature in features))
    return tuple(torch.cat([part[i] for part in parts]).to(device) for i in range(3))


@torch.no_grad()
def retrieve_poses(bank, encoder, features, metadata, history, top_k, prefer_same_scene):
    queries = encoder.encode_features(*features)
    result = bank.retrieve(queries, top_k, metadata, exclude_same_sequence=False,
                           prefer_same_scene=prefer_same_scene)
    return bank.gather_pose(result['retrieved_ids'].cpu(), history).to(queries.device), result['candidate_valid_mask'].to(queries.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', required=True)
    parser.add_argument('--retriever_data', required=True)
    parser.add_argument('--refiner_data', required=True)
    parser.add_argument('--baseline_cache', required=True)
    parser.add_argument('--baseline_checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--retriever_epochs', type=int, default=100)
    parser.add_argument('--target_temperature', type=float, default=.05)
    parser.add_argument('--key_mode', choices=['pose', 'gaze_scene'], default='pose')
    parser.add_argument('--scene_hist', action='store_true')
    parser.add_argument('--refiner_epochs', type=int, default=100)
    parser.add_argument('--refiner_batch_size', type=int, default=64)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--prefer_same_scene', action='store_true')
    parser.add_argument('--selection_endpoint_weight', type=float, default=.5)
    parser.add_argument('--temporal_refiner', action='store_true')
    parser.add_argument('--forecast_rerank', action='store_true')
    parser.add_argument('--rerank_topk', type=int, default=3)
    parser.add_argument('--align_yaw', action='store_true')
    parser.add_argument('--trust_refiner', action='store_true')
    parser.add_argument('--joint_trust_refiner', action='store_true')
    parser.add_argument('--interpolation_refiner', action='store_true')
    parser.add_argument('--terminal_refiner', action='store_true')
    parser.add_argument('--local_refiner', action='store_true')
    parser.add_argument('--endpoint_loss_weight', type=float, default=.5)
    parser.add_argument('--refiner_holdout_fraction', type=float, default=0.)
    parser.add_argument('--teacher_cache', type=Path)
    parser.add_argument('--teacher_weight', type=float, default=0.)
    args = parser.parse_args()
    if sum((args.temporal_refiner, args.align_yaw, args.trust_refiner, args.joint_trust_refiner, args.interpolation_refiner, args.terminal_refiner, args.local_refiner)) > 1:
        parser.error('Choose only one endpoint refiner variant')
    if args.target_temperature <= 0:
        parser.error('--target_temperature must be positive')
    if not 0 <= args.refiner_holdout_fraction < 1:
        parser.error('--refiner_holdout_fraction must be in [0,1)')
    if args.teacher_weight < 0 or (args.teacher_weight and args.teacher_cache is None):
        parser.error('Positive --teacher_weight requires --teacher_cache')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    bank = EndpointPoseBank.load(args.bank)
    if args.key_mode == 'gaze_scene' and any(item.dataset_name != 'GIMO' for item in bank.items):
        raise ValueError('Gaze/scene bank keys require paired GIMO modalities for every candidate')
    if bank.manifest['sample_points'] != 32768:
        raise ValueError('Current PMD evaluation requires 32768 scene points')
    train = load_split(args.retriever_data, 'train')
    gimo = [item for item in bank.items if item.dataset_name == 'GIMO']
    if [item.sample_id for item in gimo] != [item['sample_id'] for item in train['metadata']]:
        raise ValueError('GIMO query windows must match the bank before AMASS candidates')
    encoder = MultiModalRetriever(use_motion=False, use_gaze=True, use_scene=True,
                                  scene_focus=True, scene_hist=args.scene_hist, dropout=0).to(device)
    pose_encoder = EndpointPoseEncoder().to(device) if args.key_mode == 'pose' else None
    train_features = query_features(encoder, train, device)
    bank_poses = torch.stack([item.pose_joints for item in bank.items]).to(device)
    targets = bank_poses[:len(train['metadata'])]
    # Candidate pose distance supplies supervision; AMASS adds pose candidates, never fake gaze/scene.
    distance = torch.cdist(targets.flatten(1), bank_poses.flatten(1)) / 23 ** .5
    legal = torch.stack([bank.legal_candidate_mask(meta, exclude_same_sequence=False)
                         for meta in train['metadata']]).to(device)
    if not legal.any(-1).all():
        raise ValueError('Every GIMO training query needs a legal candidate')
    parameters = list(encoder.parameters()) + (list(pose_encoder.parameters()) if pose_encoder is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=3e-4)
    best_loss = float('inf')
    best_query = best_pose = None
    for epoch in range(args.retriever_epochs):
        encoder.train()
        if pose_encoder is not None:
            pose_encoder.train()
        query = encoder.encode_features(*train_features)
        keys = pose_encoder(bank_poses) if pose_encoder is not None else query
        similarity = query @ keys.T
        soft_target = torch.softmax((-distance / args.target_temperature).masked_fill(~legal, -torch.inf), -1)
        log_prediction = torch.log_softmax((similarity / .1).masked_fill(~legal, -torch.inf), -1)
        loss = -(soft_target * log_prediction.masked_fill(~legal, 0)).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_query = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            best_pose = ({k: v.detach().cpu().clone() for k, v in pose_encoder.state_dict().items()}
                         if pose_encoder is not None else None)
        if epoch % 10 == 0:
            print(json.dumps({'stage': 'retriever', 'epoch': epoch, 'loss': float(loss)}), flush=True)
    encoder.load_state_dict(best_query)
    encoder.eval()
    if pose_encoder is not None:
        pose_encoder.load_state_dict(best_pose)
        pose_encoder.eval()
    with torch.no_grad():
        bank.keys = (pose_encoder(bank_poses) if pose_encoder is not None
                     else encoder.encode_features(*train_features)).cpu()
    bank.encoder_state_dict = best_query
    bank.pose_encoder_state_dict = best_pose
    bank.manifest.update(encoder_version=encoder.encoder_version, embedding_dim=encoder.embedding_dim,
                         key_hash=bank.encoder_hash(), retriever_trained=True,
                         key_source='terminal_pose' if pose_encoder is not None else 'gaze_scene',
                         retriever_objective='terminal-pose-distance ranking',
                         target_temperature=args.target_temperature, selection_split='train ranking; valid refiner')
    bank_path = output / 'trained.bank.pt'
    bank.save(bank_path)

    features = {}
    train_metadata = None
    for split in ('train', 'valid'):
        data = load_split(args.refiner_data, split)
        if split == 'train':
            train_metadata = data['metadata']
        cached = torch.load(Path(args.baseline_cache) / f'{split}.pt', map_location='cpu')
        if data['metadata'] != cached['metadata']:
            raise ValueError(f'{split} PMD forecast cache does not match data')
        query = query_features(encoder, data, device)
        poses, valid = retrieve_poses(bank, encoder, query, data['metadata'], data['joints_input'],
                                      args.top_k, args.prefer_same_scene)
        features[split] = {'base': cached['prediction'][:, 6:].to(device),
                           'target': data['joints_label'].to(device), 'poses': poses, 'valid': valid,
                           'history': data['joints_input'].to(device),
                           'gazes': data['gazes'].to(device),
                           'query_context': encoder.encode_features(*query).detach()}
        if args.teacher_cache is not None:
            teacher = torch.load(args.teacher_cache / f'{split}.pt', map_location='cpu')
            if teacher['metadata'] != data['metadata']:
                raise ValueError(f'{split} teacher cache does not match data')
            features[split]['teacher'] = teacher['prediction'].to(device)
    if args.refiner_holdout_fraction:
        recordings = sorted({row['sequence_id'] for row in train_metadata})
        random.Random(args.seed).shuffle(recordings)
        count = max(1, round(len(recordings) * args.refiner_holdout_fraction))
        held_out = set(recordings[:count])
        valid_ids = torch.tensor([n for n, row in enumerate(train_metadata)
                                  if row['sequence_id'] in held_out], device=device)
        train_ids = torch.tensor([n for n, row in enumerate(train_metadata)
                                  if row['sequence_id'] not in held_out], device=device)
        if not len(valid_ids) or not len(train_ids):
            raise ValueError('Recording holdout produced an empty refiner split')
        source = features['train']
        features['valid'] = {key: value[valid_ids] for key, value in source.items()}
        features['train'] = {key: value[train_ids] for key, value in source.items()}
        (output / 'refiner_holdout.json').write_text(json.dumps({
            'recordings': sorted(held_out), 'train_windows': len(train_ids),
            'valid_windows': len(valid_ids)}, indent=2) + '\n')
    refiner = (EndpointLocalMotionRefiner() if args.local_refiner else
               EndpointTerminalRefiner() if args.terminal_refiner else
               EndpointInterpolationRefiner() if args.interpolation_refiner else
               EndpointJointTrustRefiner() if args.joint_trust_refiner else
               EndpointTrustRefiner() if args.trust_refiner else
               EndpointPoseTemporalRefiner(rerank_topk=args.rerank_topk if args.forecast_rerank else 0)
               if args.temporal_refiner else EndpointPoseRefiner(rerank_topk=args.rerank_topk if args.forecast_rerank else 0,
                                                                 align_yaw=args.align_yaw)).to(device)
    if args.trust_refiner or args.joint_trust_refiner:
        with torch.no_grad():
            f = features['train']
            observed, _, _ = refiner.observation_features(f['base'], f['poses'], f['valid'], f['history'])
            observed = observed.reshape(-1, observed.shape[-1])
            refiner.feature_mean.copy_(observed.mean(0))
            refiner.feature_std.copy_(observed.std(0).clamp_min(.05))
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=1e-3, weight_decay=1e-3)
    best_score = float('inf')
    best_refiner = None
    for epoch in range(args.refiner_epochs):
        refiner.train()
        f = features['train']
        batches = torch.randperm(len(f['base']), device=device).split(args.refiner_batch_size)
        for ids in batches:
            prediction, delta = refiner(f['base'][ids], f['poses'][ids], f['valid'][ids], f['history'][ids],
                                        **({'gazes': f['gazes'][ids], 'query_context': f['query_context'][ids]}
                                           if args.terminal_refiner or args.local_refiner else {}))
            relative = prediction - prediction[:, :, :1]
            target = f['target'][ids] - f['target'][ids, :, :1]
            joint_error = (relative - target).norm(dim=-1)
            loss = joint_error.mean() + args.endpoint_loss_weight * joint_error[:, -1].mean() + .1 * delta.square().mean()
            if args.teacher_weight:
                teacher = f['teacher'][ids] - f['teacher'][ids, :, :1]
                distillation = (relative - teacher).norm(dim=-1)
                loss = loss + args.teacher_weight * (distillation.mean() +
                                                     args.endpoint_loss_weight * distillation[:, -1].mean())
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        refiner.eval()
        with torch.no_grad():
            v = features['valid']
            valid_pred, _ = refiner(v['base'], v['poses'], v['valid'], v['history'],
                                    **({'gazes': v['gazes'], 'query_context': v['query_context']}
                                       if args.terminal_refiner or args.local_refiner else {}))
            metrics = dict(zip(METRICS, errors(valid_pred, v['target']).mean(0).tolist()))
        score = metrics['mpjpe_mm'] + args.selection_endpoint_weight * metrics['endpoint_mpjpe_mm']
        if score < best_score:
            best_score = score
            best_refiner = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}
            (output / 'best_valid_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        if epoch % 10 == 0:
            print(json.dumps({'stage': 'refiner', 'epoch': epoch, 'loss': float(loss), **metrics}), flush=True)
        if epoch >= 30 and score > best_score + 5:
            break
    torch.save(best_refiner, output / 'best_refiner.pt')
    from config.config import MotionFromGazeConfig
    from model.sif3d_mod import SIF3D
    from human_body_prior.tools.model_loader import load_vposer
    import smplx

    baseline_config = torch.load(Path(args.baseline_cache) / 'train.pt', map_location='cpu')['config']
    config_args = ['--use_rag', '--rag_bank', str(bank_path), '--rag_modalities', 'gaze+scene',
                   '--rag_refiner_type', 'pose_endpoint_local' if args.local_refiner else 'pose_endpoint_terminal' if args.terminal_refiner else 'pose_endpoint_interpolation' if args.interpolation_refiner else 'pose_endpoint_joint_trust' if args.joint_trust_refiner else 'pose_endpoint_trust' if args.trust_refiner else 'pose_endpoint_temporal' if args.temporal_refiner else 'pose_endpoint_aligned' if args.align_yaw else 'pose_endpoint', '--rag_train_stage', 'refiner',
                   '--rag_allow_same_sequence', '--rag_top_k', str(args.top_k),
                   '--rag_endpoint_topk', str(args.rerank_topk),
                   '--sample_points', str(bank.manifest['sample_points']),
                   '--smplx_path', baseline_config['smplx_path'],
                   '--vposer_path', baseline_config['vposer_path'],
                   '--traj_ckpts', baseline_config['traj_ckpts']]
    if baseline_config['occ']:
        config_args.append('--occ')
    if args.prefer_same_scene:
        config_args.append('--rag_prefer_same_scene')
    if args.forecast_rerank:
        config_args.append('--rag_forecast_rerank')
    config = MotionFromGazeConfig().parse_args(config_args)
    vposer, _ = load_vposer(config.vposer_path, vp_model='snapshot')
    body = smplx.create(config.smplx_path, model_type='smplx', gender='neutral', ext='npz',
                        num_pca_comps=12, batch_size=16)
    model = SIF3D(config, vposer, body)
    baseline_weights = torch.load(args.baseline_checkpoint, map_location='cpu')
    model.load_prediction_checkpoint(baseline_weights, allow_baseline=True)
    model.motion_refiner.load_state_dict(best_refiner)
    torch.save({key: value.cpu() for key, value in model.state_dict().items()
                if not key.startswith('smplx_model.')}, output / 'best.pth')
    checked = [key for key in baseline_weights if key in model.state_dict() and not key.startswith('smplx_model.')]
    if any(not torch.equal(baseline_weights[key].cpu(), model.state_dict()[key].cpu()) for key in checked):
        raise ValueError('Frozen PMD backbone changed during endpoint RAG training')
    (output / 'model_args.json').write_text(json.dumps(config_args, indent=2) + '\n')
    (output / 'config.json').write_text(json.dumps({**vars(args), 'trained_bank': str(bank_path)},
                                                   indent=2, default=str) + '\n')
    print(f'Trained bank: {bank_path}; best validation: {json.loads((output / "best_valid_metrics.json").read_text())}', flush=True)


if __name__ == '__main__':
    main()
