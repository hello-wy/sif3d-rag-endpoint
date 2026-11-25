"""Extend a pooled encoder with observed last-frame/change summaries, preserving initialization."""
import argparse,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.multimodal_rag import MotionBank,MultiModalRetriever,retriever_from_bank


def upgrade_encoder(encoder):
    if encoder.temporal or encoder.rich_summary:
        raise ValueError('Initialization upgrade expects an ordinary pooled encoder')
    upgraded=MultiModalRetriever(motion_dim=encoder.motion_dim,joint_count=encoder.joint_count,
        embedding_dim=encoder.embedding_dim,use_motion=encoder.use_motion,use_gaze=encoder.use_gaze,
        use_scene=encoder.use_scene,dropout=0.0,rich_summary=True)
    old=encoder.state_dict();new=upgraded.state_dict()
    for key,value in old.items():
        if key in ('motion_branch.0.weight','gaze_branch.0.weight'):
            new[key].zero_();new[key][:,:value.shape[1]].copy_(value)
        else:new[key].copy_(value)
    upgraded.load_state_dict(new,strict=True)
    return upgraded


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);bank=MotionBank.load(a.bank);model=upgrade_encoder(retriever_from_bank(bank)).eval()
    bank.encode_keys(model,batch_size=8);bank.manifest.update(retriever_trained=False,
        initialization='v1 weights plus zero new last-frame/change columns; requires retraining',encoder_version=model.encoder_version)
    bank.save(a.output);print('saved',len(bank),'windows',flush=True)

if __name__=='__main__':main()
