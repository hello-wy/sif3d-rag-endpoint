"""Use aligned cached rows; reserve training recordings for RAG validation."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataroot',default='/data/wuyang/human/GIMO')
    p.add_argument('--dataset_csv',default='dataset_with_motion_label_copy')
    p.add_argument('--folder',default='318_4096_single')
    p.add_argument('--sample_points',type=int,default=32768)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    root=Path(a.dataroot);out=Path(a.output)
    df=pd.read_csv(root/(a.dataset_csv+'.csv'))
    train=df[df.training==1].reset_index(drop=True);test=df[df.training==0].reset_index(drop=True)
    if train.duplicated(['sequence_path','start_frame']).any():
        raise ValueError('Source manifest must be deduplicated and aligned first')
    test_intervals={}
    for _,row in test.iterrows():test_intervals.setdefault(str(row.sequence_path),[]).append((int(row.start_frame),int(row.start_frame)+225))
    legal=[];removed=[]
    for i,row in train.iterrows():
        start=int(row.start_frame)
        if any(max(start,l)<=min(start+225,r) for l,r in test_intervals.get(str(row.sequence_path),[])):removed.append(i)
        else:legal.append(i)
    # Avoid taking validation from recordings represented in the test split.
    available=sorted(set(train.iloc[legal].sequence_path)-set(test.sequence_path))
    if len(available)<5:raise ValueError('Not enough independent recordings for RAG validation')
    generator=torch.Generator().manual_seed(a.seed)
    chosen={available[i] for i in torch.randperm(len(available),generator=generator)[:max(4,len(available)//5)].tolist()}
    indices={'train':[i for i in legal if train.iloc[i].sequence_path not in chosen],
             'valid':[i for i in legal if train.iloc[i].sequence_path in chosen], 'test':list(range(len(test)))}
    caches={}
    names=['gazes','poses_input','poses_label','joints_input','joints_label','occ_map',f'scene_points_{a.sample_points}']
    for split in ['train','test']:
        caches[split]={}
        for name in names:
            value=torch.as_tensor(torch.load(root/a.folder/split/(name+'.pth'),map_location='cpu'))
            if len(value)!=len(train if split=='train' else test):raise ValueError('CSV/cache mismatch')
            if not torch.isfinite(value).all():raise ValueError('Nonfinite cache')
            caches[split][name]=value
    out.mkdir(parents=True,exist_ok=False)
    for split,ids in indices.items():
        source='test' if split=='test' else 'train'
        directory=out/'cache'/split;directory.mkdir(parents=True)
        for name,value in caches[source].items():torch.save(value[ids].clone(),directory/(name+'.pth'))
    train.iloc[indices['train']].to_csv(out/'train.csv',index=False)
    train.iloc[indices['valid']].to_csv(out/'valid.csv',index=False)
    test.to_csv(out/'test.csv',index=False)
    pd.concat([train.iloc[indices['train']],test],ignore_index=True).to_csv(out/'dataset.csv',index=False)
    manifest={'source_csv':str(root/(a.dataset_csv+'.csv')),'source_csv_sha256':hashlib.sha256((root/(a.dataset_csv+'.csv')).read_bytes()).hexdigest(),
              'source_cache':str(root/a.folder),'retained_source_indices':indices,'test_overlap_removed_source_indices':removed,
              'valid_sequences':sorted(chosen),'samples':{s:len(ids) for s,ids in indices.items()},'seed':a.seed,
              'limitation':'RAG validation recordings withheld from RAG fitting; baseline predictor previously trained on original training recordings.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest['samples']))

if __name__=='__main__':main()
