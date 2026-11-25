"""Evaluate a bank on the isolated validation/test cache, excluding query recordings."""
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.multimodal_rag import MotionBank,retriever_from_bank
from scripts.rag_data import load_split

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--data',required=True)
    p.add_argument('--split',choices=['valid','test'],default='test');p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    bank=MotionBank.load(a.bank);model=retriever_from_bank(bank).eval();data=load_split(a.data,a.split);records=[]
    with torch.no_grad():
        for i,metadata in enumerate(data['metadata']):
            history=data['joints_input'][i:i+1]
            query=model(data['poses_input'][i:i+1],history,data['gazes'][i:i+1],data['scenes'][i:i+1])
            result=bank.retrieve(query,5,[metadata]);valid=result['candidate_valid_mask'][0]
            if not valid.any():raise ValueError('No legal candidates')
            future=bank.gather_future(result['retrieved_ids'],history)[0,valid]
            relative=future-future[:,:,:1];target=data['joints_label'][i]-data['joints_label'][i,:,:1]
            e=(relative-target).norm(dim=-1).mean((1,2))*1000
            records.append({'sample_id':metadata['sample_id'],'top1_mpjpe_mm':float(e[0]),'oracle5_mm':float(e.min()),
                            'retrieved_ids':result['retrieved_ids'][0].tolist()})
    report={'split':a.split,'queries':len(records),'top1_mpjpe_mm':sum(r['top1_mpjpe_mm'] for r in records)/len(records),
            'oracle5_mm':sum(r['oracle5_mm'] for r in records)/len(records),'per_sample':records,
            'note':'Future ground truth used for diagnostics only; no future in the query.'}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='per_sample'})
