"""Fine-tune an independent endpoint orientation predictor; translation stays outside this branch."""
import argparse,copy,json,random,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config.config import MotionFromGazeConfig
from model.sif3d_mod import TrajPredictor
from scripts.rag_data import load_split
from utils.rotation import axis_angle_to_matrix,matrix_to_axis_angle


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=40);p.add_argument('--patience',type=int,default=10)
    p.add_argument('--stage',choices=['head','tia','all'],default='tia');p.add_argument('--consistency',type=float,default=.1)
    p.add_argument('--no_augmentation',action='store_true')
    a,remaining=p.parse_known_args();c=MotionFromGazeConfig().parse_args(remaining);torch.set_num_threads(4);torch.manual_seed(c.seed);np.random.seed(c.seed);random.seed(c.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu');state={k.removeprefix('traj_predictor.'):v for k,v in state.items() if k.startswith('traj_predictor.')}
    model=TrajPredictor(c).to(device);model.load_state_dict(state,strict=True);teacher=copy.deepcopy(model).eval().requires_grad_(False)
    if a.stage!='all':
        model.requires_grad_(False);model.trajectory_fc.requires_grad_(True)
        if a.stage=='tia':model.tia_blocks.requires_grad_(True)
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=c.lr,weight_decay=1e-3)
    data={s:load_split(a.data,s) for s in ['train','valid']};best=float('inf');best_epoch=-1
    (out/'config.json').write_text(json.dumps({'model':vars(c),'training':vars(a)},indent=2)+'\n')
    for epoch in range(a.epochs):
        model.eval();model.trajectory_fc.train()
        if a.stage=='tia':model.tia_blocks.train()
        if a.stage=='all':model.train()
        order=torch.randperm(len(data['train']['metadata']));losses=[]
        for start in range(0,len(order),c.batch_size):
            ids=order[start:start+c.batch_size];d=data['train'];motion=d['poses_input'][ids].to(device).clone();joints=d['joints_input'][ids].to(device);scene=d['scenes'][ids].to(device);gaze=d['gazes'][ids].to(device);occ=d['occ_map'][ids].to(device);target=d['poses_label'][ids,:,:3].to(device)
            if len(ids)<2:continue
            # Apply one common yaw rotation per batch; rotate all observed spatial modalities.
            turns=0 if a.no_augmentation else int(torch.randint(0,4,()).item());yaw=turns*torch.pi/2;R=axis_angle_to_matrix(motion.new_tensor([0.,yaw,0.]));motion[...,:3]=matrix_to_axis_angle(R@axis_angle_to_matrix(motion[...,:3]));motion[...,3:6]=motion[...,3:6]@R.T;joints=joints@R.T;scene=scene@R.T;gaze=gaze@R.T;target_rotation=R@axis_angle_to_matrix(target)
            occ=torch.rot90(occ,-turns,dims=(-2,-1)).contiguous()
            ori,trans=model(motion,joints,scene,gaze,occ);rotation=axis_angle_to_matrix(ori[:,6:]);error=(rotation-target_rotation).square().mean((-1,-2))
            with torch.no_grad():old_ori,old_trans=teacher(motion,joints,scene,gaze,occ);old_rotation=axis_angle_to_matrix(old_ori[:,6:])
            consistency=(rotation-old_rotation).square().mean()
            loss=error.mean()+3*error[:,-1].mean()+a.consistency*consistency
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
        model.eval();end_errors=[];all_errors=[]
        with torch.no_grad():
            d=data['valid']
            for i in range(len(d['metadata'])):
                ori,_=model(d['poses_input'][i:i+1].to(device),d['joints_input'][i:i+1].to(device),d['scenes'][i:i+1].to(device),d['gazes'][i:i+1].to(device),d['occ_map'][i:i+1].to(device));rotation=axis_angle_to_matrix(ori[:,6:]);target=axis_angle_to_matrix(d['poses_label'][i:i+1,:,:3].to(device));relative=rotation.transpose(-1,-2)@target;angle=torch.acos(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1+1e-6,1-1e-6))*180/torch.pi;all_errors.append(float(angle.mean()));end_errors.append(float(angle[:,-1].mean()))
        result={'epoch':epoch,'train_loss':sum(losses)/len(losses),'valid_orientation_deg':sum(all_errors)/len(all_errors),'valid_endpoint_orientation_deg':sum(end_errors)/len(end_errors)}
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        print(json.dumps(result),flush=True);score=result['valid_endpoint_orientation_deg']
        if score<best:
            best=score;best_epoch=epoch;torch.save(model.state_dict(),out/'best.pth');(out/'best_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        if epoch-best_epoch>=a.patience:break
    torch.save(model.state_dict(),out/'last.pth');(out/'summary.json').write_text(json.dumps({'epochs':epoch+1,'best_epoch':best_epoch,'best_valid_endpoint_orientation_deg':best},indent=2)+'\n')

if __name__=='__main__':main()
