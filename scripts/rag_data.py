"""Shared cache loading and full-window metrics for isolated RAG experiments."""
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import torch
from model.multimodal_rag import MotionBank


def load_split(root, split, sample_points=32768):
    root=Path(root);rows=pd.read_csv(root/(split+'.csv'))
    names=['gazes','poses_input','poses_label','joints_input','joints_label','occ_map',f'scene_points_{sample_points}']
    data={k:torch.as_tensor(torch.load(root/'cache'/split/(k+'.pth'),map_location='cpu')).float() for k in names}
    if any(len(v)!=len(rows) for v in data.values()):raise ValueError('Cache manifest count mismatch')
    data['scenes']=data.pop(f'scene_points_{sample_points}')
    data['joints_input']=data['joints_input'][:,:,:23].clone()
    data['joints_label']=data['joints_label'][:,:,:23].clone()
    data['metadata']=[{'sequence_id':str(row.sequence_path),'scene_id':str(row.scene),'sample_id':f'{row.sequence_path}:{int(row.start_frame)}'} for _,row in rows.iterrows()]
    data['rows']=rows
    return data


def make_bank(data):
    dataset=SimpleNamespace(train=True,sequences_path_list=data['rows'].sequence_path.tolist(),
       scenes_path_list=data['rows'].scene.tolist(),start_end_list=list(zip(data['rows'].start_frame,data['rows'].end_frame)),
       input_seq_len=6,output_seq_len=10,fps=2,poses_input=data['poses_input'],poses_label=data['poses_label'],
       joints_input=data['joints_input'],joints_label=data['joints_label'],gazes=data['gazes'],scenes=data['scenes'])
    class CachedDataset:
        def __len__(self):return len(data['rows'])
        def __getattr__(self,name):return getattr(dataset,name)
    return MotionBank.from_gimo_dataset(CachedDataset())


def errors(pred,target):
    translation=(pred[:,:,0]-target[:,:,0]).norm(dim=-1)*1000
    relative=(pred-pred[:,:,:1])-(target-target[:,:,:1])
    error=relative.norm(dim=-1)*1000
    return torch.stack([translation.mean(1),translation[:,-1],error.mean((1,2)),error[:,-1].mean(1)],-1)

METRICS=['translation_mm','endpoint_translation_mm','mpjpe_mm','endpoint_mpjpe_mm']
