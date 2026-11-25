"""Construct matched pooled/temporal retrieval banks from an isolated train split."""
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.rag_data import load_split,make_bank
from model.multimodal_rag import MultiModalRetriever

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();torch.set_num_threads(4);data=load_split(a.data,'train')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    for name,temporal in [('pooled',False),('temporal',True)]:
        torch.manual_seed(a.seed);bank=make_bank(data);encoder=MultiModalRetriever(temporal=temporal,dropout=0.0)
        bank.encode_keys(encoder)
        bank.manifest.update(encoder_version=encoder.encoder_version,use_motion=True,use_gaze=True,use_scene=True,
                             input_seq_len=6,output_seq_len=10,fps=2,sample_points=32768,retriever_trained=False,
                             validation_manifest=json.loads((Path(a.data)/'manifest.json').read_text()))
        bank.save(root/(name+'_initial.pt'));print(name,len(bank),flush=True)
