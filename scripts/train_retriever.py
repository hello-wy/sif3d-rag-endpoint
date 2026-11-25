"""Train the differentiable sampled-candidate retrieval objective."""

import argparse
import random
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.multimodal_rag import MotionBank, align_future_to_query, retrieval_soft_label_loss, retriever_from_bank,reference_frame
from scripts.rag_data import load_split


def _batch(items, indices, device):
    return (
        torch.stack([items[i].history_motion for i in indices]).to(device),
        torch.stack([items[i].history_joints for i in indices]).to(device),
        torch.stack([items[i].history_gaze for i in indices]).to(device),
        torch.stack([items[i].scene for i in indices]).to(device),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--bank', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--bank_output', default=None, help='trained bank with rebuilt keys; defaults to <output>.bank.pt')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--candidates', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--validation_data', default=None, help='Recording-held-out RAG validation cache root')
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.candidates) <= 0:
        p.error('epochs, batch_size and candidates must be positive')
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    bank = MotionBank.load(args.bank)
    if args.bank_output:
        Path(args.bank_output).parent.mkdir(parents=True, exist_ok=True)
    if len(bank) < 2:
        raise ValueError('retriever training needs at least two train windows')
    device = torch.device(args.device)
    model = retriever_from_bank(bank, dropout=0.1).to(device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Coordinate summaries are input-only and fixed; cache them once instead of
    # rotating dense scenes and synchronizing candidate hips on every GPU step.
    feature_parts = []
    with torch.no_grad():
        for start in range(0,len(bank),8):
            feature_parts.append(model.prepare_features(*_batch(bank.items,list(range(start,min(start+8,len(bank)))) ,torch.device('cpu'))))
        cached_features = tuple(torch.cat([part[i] for part in feature_parts]).to(device) for i in range(3))
        canonical_future = torch.stack([(item.future_joints-item.future_joints[:,:1]) @ reference_frame(item.history_joints)[1] for item in bank.items]).to(device)
    all_ids = list(range(len(bank)))
    validation = load_split(args.validation_data, 'valid') if args.validation_data else None
    best = float('inf'); best_state = None
    validation_errors = None
    if validation is not None:
        validation_errors = []
        for qi in range(len(validation['metadata'])):
            history = validation['joints_input'][qi]
            target = validation['joints_label'][qi]
            validation_errors.append(torch.stack([
                torch.linalg.vector_norm((aligned - aligned[:, :1]) - (target - target[:, :1]), dim=-1).mean()
                for aligned in [align_future_to_query(item.future_joints, item.history_joints, history) for item in bank.items]]))
        validation_errors = torch.stack(validation_errors).to(device)
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(all_ids)
        losses = []
        for start in range(0, len(all_ids), args.batch_size):
            qids = all_ids[start:start + args.batch_size]
            candidate_ids = []
            legal_by_query = [[i for i in all_ids if bank.items[i].sequence_id != bank.items[qi].sequence_id] for qi in qids]
            count = min(args.candidates, min(len(ids) for ids in legal_by_query))
            if count == 0:
                raise ValueError('No legal candidates outside the query recording')
            for qi, legal in zip(qids, legal_by_query):
                candidate_ids.append(random.sample(legal, count))
            candidate_tensor = torch.tensor(candidate_ids, device=device)
            query_tensor = torch.tensor(qids, device=device)
            q_emb = model.encode_features(*[value[query_tensor] for value in cached_features])
            flat = candidate_tensor.flatten()
            c_emb = model.encode_features(*[value[flat] for value in cached_features]).view(len(qids),count,-1)
            target_distance = (canonical_future[candidate_tensor]-canonical_future[query_tensor,None]).norm(dim=-1).mean(dim=(-1,-2))
            # The targets use only stored training futures; query gradients are
            # retained through the normalized similarity below.
            loss = retrieval_soft_label_loss(torch.einsum('bd,bkd->bk', q_emb, c_emb), target_distance)
            retrieval_loss = loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(retrieval_loss.detach()))
        metrics = {'epoch': epoch, 'retrieval_kl': sum(losses) / max(len(losses), 1),
                   'selection_split': 'valid' if validation is not None else 'last'}
        if validation is not None:
            bank.encode_keys(model, device=device)
            with torch.no_grad():
                query = model(validation['poses_input'].to(device), validation['joints_input'].to(device),
                              validation['gazes'].to(device), validation['scenes'].to(device))
                result = bank.retrieve(query, top_k=5, metadata=validation['metadata'])
                ids = result['retrieved_ids']
                if not result['candidate_valid_mask'].all():
                    raise ValueError('Validation requires five legal distinct candidates per query')
                distance = validation_errors.gather(1, ids)
                metrics['valid_top1_future_mpjpe_mm'] = float(distance[:, 0].mean()*1000)
                metrics['valid_oracle5_mm'] = float(distance.min(1).values.mean()*1000)
                if metrics['valid_top1_future_mpjpe_mm'] < best:
                    best = metrics['valid_top1_future_mpjpe_mm']
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(metrics), flush=True)
        with open(args.output + '.metrics.jsonl', 'a') as handle:
            handle.write(json.dumps(metrics) + '\n')
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    checkpoint = {'state_dict': model.state_dict(), 'encoder_version': model.encoder_version, 'embedding_dim': model.embedding_dim}
    torch.save(checkpoint, args.output)
    bank.manifest.update(retriever_trained=True, retriever_epochs=args.epochs, retriever_seed=args.seed,
                         selection_split='valid' if validation is not None else 'last', valid_top1_mpjpe_mm=best if best_state is not None else None, retriever_objective='future-distance KL; no query label input')
    bank.encode_keys(model)
    bank.save(args.bank_output or (args.output + '.bank.pt'))


if __name__ == '__main__':
    main()
