"""Retrieval-only Top-1 and candidate-set Oracle@K diagnostics.

Oracle@K here selects the lowest stored future distance among the returned
candidate set. It is a diagnostic, not an inference result or a full-bank bound.
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.multimodal_rag import MotionBank, MultiModalRetriever, align_future_to_query


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--bank', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--top_k', type=int, default=5)
    args = p.parse_args()
    bank = MotionBank.load(args.bank)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model = MultiModalRetriever(
        embedding_dim=int(ckpt.get('embedding_dim', bank.manifest.get('embedding_dim', 256))),
        use_motion=bool(bank.manifest.get('use_motion', True)),
        use_gaze=bool(bank.manifest.get('use_gaze', True)),
        use_scene=bool(bank.manifest.get('use_scene', True)),
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    bank.encode_keys(model)
    top1, oracle = [], []
    with torch.no_grad():
        for qid, query in enumerate(bank.items):
            query_emb = model(query.history_motion[None], query.history_joints[None], query.history_gaze[None], query.scene[None])
            result = bank.retrieve(query_emb, top_k=args.top_k, metadata=[query.metadata()], exclude_same_sequence=True)
            ids = result['retrieved_ids'][0]
            distances = []
            for idx in ids.tolist():
                if idx < 0:
                    continue
                item = bank.items[idx]
                aligned = align_future_to_query(item.future_joints, item.history_joints, query.history_joints)
                qrel = query.future_joints - query.future_joints[:, :1]
                prel = aligned - aligned[:, :1]
                distances.append(float(torch.linalg.vector_norm(prel - qrel, dim=-1).mean()))
            if distances:
                top1.append(distances[0])
                oracle.append(min(distances))
    print({'queries': len(top1), 'top1_future_mpjpe': sum(top1) / max(len(top1), 1), 'oracle_at_k': sum(oracle) / max(len(oracle), 1), 'k': args.top_k})


if __name__ == '__main__':
    main()
