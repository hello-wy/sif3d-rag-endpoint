"""Train terminal-pose guidance inside frozen PMD while preserving its trajectory."""

import argparse
import json
import sys
from pathlib import Path

import smplx
import torch
from human_body_prior.tools.model_loader import load_vposer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.config import MotionFromGazeConfig
from model.sif3d_mod import SIF3D
from scripts.rag_data import load_split, errors, METRICS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--initial_checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--endpoint_loss_weight', type=float, default=2.)
    parser.add_argument('--root_loss_weight', type=float, default=1000.)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.root_loss_weight < 0:
        parser.error('Invalid training parameters')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    config_args = ['--use_rag', '--rag_bank', args.bank, '--rag_modalities', 'gaze+scene',
                   '--rag_refiner_type', 'pose_endpoint_interpolation', '--rag_endpoint_adapter',
                   '--rag_allow_same_sequence', '--rag_prefer_same_scene', '--rag_top_k', '3',
                   '--rag_train_stage', 'adapter', '--sample_points', '32768', '--occ',
                   '--smplx_path', '/home/customer/sif3d-raw/data',
                   '--vposer_path', '/home/customer/sif3d-raw/vposer_v1_0']
    config = MotionFromGazeConfig().parse_args(config_args)
    vposer, _ = load_vposer(config.vposer_path, vp_model='snapshot')
    body = smplx.create(config.smplx_path, model_type='smplx', gender='neutral', ext='npz',
                        num_pca_comps=12, batch_size=args.batch_size * 16).to(device)
    model = SIF3D(config, vposer.to(device), body).to(device)
    initial = torch.load(args.initial_checkpoint, map_location='cpu')
    model.load_prediction_checkpoint(initial)
    model.motion_refiner.mix_logit.requires_grad_(False)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-3)
    data = {split: load_split(args.data, split) for split in ('train', 'valid')}
    for split in data:
        if any(len(value) != len(data[split]['metadata']) for key, value in data[split].items()
               if isinstance(value, torch.Tensor)):
            raise ValueError(f'{split} cache row mismatch')

    def batch(split, indices):
        source = data[split]
        real = len(indices)
        if real < args.batch_size:
            indices = torch.cat((indices, indices[-1:].expand(args.batch_size - real)))
        ids = indices.tolist()
        tensors = [source[key][indices].to(device) for key in
                   ('poses_input', 'joints_input', 'scenes', 'gazes', 'occ_map')]
        labels = source['joints_label'][indices].to(device)
        metadata = [source['metadata'][index] for index in ids]
        return tensors, labels, metadata, real

    def pair_forward(tensors, metadata, training):
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device)
        model.endpoint_adapter = False
        with torch.no_grad():
            baseline = model(*tensors, rag_metadata=metadata)[1][:, 6:]
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        model.endpoint_adapter = True
        if training:
            prediction = model(*tensors, rag_metadata=metadata)[1][:, 6:]
        else:
            with torch.no_grad():
                prediction = model(*tensors, rag_metadata=metadata)[1][:, 6:]
        return prediction, baseline

    def validate():
        model.eval()
        predictions, targets = [], []
        max_root_change = 0.
        for indices in torch.arange(len(data['valid']['metadata'])).split(args.batch_size):
            tensors, labels, metadata, real = batch('valid', indices)
            prediction, baseline = pair_forward(tensors, metadata, False)
            max_root_change = max(max_root_change,
                                  float((prediction[:real, :, 0] - baseline[:real, :, 0]).abs().max()) * 1000)
            predictions.append(prediction[:real].cpu())
            targets.append(labels[:real].cpu())
        values = errors(torch.cat(predictions), torch.cat(targets)).mean(0).tolist()
        return dict(zip(METRICS, values)), max_root_change

    best_score = float('inf')
    best_epoch = -1
    best_weights = None
    (output / 'config.json').write_text(json.dumps({**vars(args), 'model_args': config_args}, indent=2) + '\n')
    for epoch in range(args.epochs):
        model.train()
        losses = []
        order = torch.randperm(len(data['train']['metadata']))
        for indices in order.split(args.batch_size):
            tensors, labels, metadata, real = batch('train', indices)
            prediction, baseline = pair_forward(tensors, metadata, True)
            prediction, baseline, labels = prediction[:real], baseline[:real], labels[:real]
            relative = prediction - prediction[:, :, :1]
            target = labels - labels[:, :, :1]
            error = (relative - target).norm(dim=-1)
            root_delta = prediction[:, :, 0] - baseline[:, :, 0]
            loss = (error.mean() + args.endpoint_loss_weight * error[:, -1].mean() +
                    args.root_loss_weight * root_delta.square().mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics, root_change = validate()
        result = {'epoch': epoch, 'train_loss': sum(losses) / len(losses),
                  'root_max_coordinate_change_mm': root_change, **metrics}
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)
        score = metrics['mpjpe_mm'] + metrics['endpoint_mpjpe_mm']
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                            if not key.startswith('smplx_model.')}
            torch.save(best_weights, output / 'best.pth')
            (output / 'best_valid_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        if epoch - best_epoch >= args.patience:
            break
    changed = [key for key, value in initial.items()
               if key in best_weights and not key.startswith(('prototype_encoder.', 'prototype_adapter.',
                                                               'motion_refiner.'))
               and not torch.equal(value.cpu(), best_weights[key])]
    if changed:
        raise ValueError(f'Frozen backbone changed: {changed}')
    print(f'Selected epoch {best_epoch}; frozen backbone tensors unchanged', flush=True)


if __name__ == '__main__':
    main()
