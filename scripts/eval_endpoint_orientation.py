"""Evaluate a separate orientation branch around a frozen RAG forecast pelvis."""
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config.config import MotionFromGazeConfig
from model.sif3d_mod import TrajPredictor
from utils.rotation import axis_angle_to_matrix,matrix_to_axis_angle
from scripts.rag_data import load_split,errors,METRICS


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--base_features',required=True)
    p.add_argument('--base_cache',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(4);run=Path(a.run);saved=json.loads((run/'config.json').read_text());c=MotionFromGazeConfig().parse_args([])
    for k,v in saved['model'].items():setattr(c,k,v)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=TrajPredictor(c).to(device);model.load_state_dict(torch.load(run/'best.pth',map_location='cpu'),strict=True);model.eval();root=Path(a.base_features);cache=Path(a.base_cache);report={}
    with torch.no_grad():
        for split in ['valid','test']:
            data=load_split(saved['training']['data'],split);features=torch.load(root/(split+'_ridge_features.pt'),map_location='cpu');base=features['base'];base_cache=torch.load(cache/(split+'.pt'),map_location='cpu');old_rotation=axis_angle_to_matrix(base_cache['latent'][:,6:,:3]);rotations=[]
            for i in range(len(data['metadata'])):
                ori,_=model(data['poses_input'][i:i+1].to(device),data['joints_input'][i:i+1].to(device),data['scenes'][i:i+1].to(device),data['gazes'][i:i+1].to(device),data['occ_map'][i:i+1].to(device));rotations.append(axis_angle_to_matrix(ori[:,6:]).cpu())
            rotation=torch.cat(rotations);difference=matrix_to_axis_angle(rotation@old_rotation.transpose(-1,-2));values={}
            for strength in [0,.25,.5,.75,1]:
                ramp=torch.zeros(10);ramp[-4:]=torch.linspace(.25,1,4);correction=axis_angle_to_matrix(difference*strength*ramp[None,:,None]);relative=base-base[:,:,:1];pred=relative@correction.transpose(-1,-2)+base[:,:,:1];values[str(strength)]=dict(zip(METRICS,errors(pred,data['joints_label']).mean(0).tolist()))
                assert torch.equal(pred[:,:,0],base[:,:,0])
            report[split]=values;torch.save({'relative_rotation':difference,'new_rotation':rotation,'base':base},run/(split+'_correction.pt'));print(split,values,flush=True)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
