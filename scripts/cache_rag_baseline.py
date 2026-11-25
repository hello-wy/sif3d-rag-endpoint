"""Cache a frozen SIF3D forecast once for controlled refiner training."""
import argparse,json,sys
from pathlib import Path
import torch,smplx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config.config import MotionFromGazeConfig
from model.sif3d_mod import SIF3D
from human_body_prior.tools.model_loader import load_vposer
from scripts.rag_data import load_split,errors,METRICS


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True)
    args,remaining=p.parse_known_args();config=MotionFromGazeConfig().parse_args(remaining)
    torch.set_num_threads(4);torch.manual_seed(config.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vposer,_=load_vposer(config.vposer_path,vp_model='snapshot')
    body=smplx.create(config.smplx_path,model_type='smplx',gender='neutral',ext='npz',num_pca_comps=12,batch_size=16).to(device)
    model=SIF3D(config,vposer.to(device),body).to(device).eval()
    model.load_prediction_checkpoint(torch.load(args.checkpoint,map_location='cpu'))
    trajectory_hidden = []
    hook = model.traj_predictor.trajectory_fc.register_forward_pre_hook(lambda module, inputs: trajectory_hidden.append(inputs[0].detach().cpu()))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for split in ['train','valid','test']:
        data=load_split(args.data,split);predictions=[];hiddens=[];latents=[];trajectory_hidden.clear()
        with torch.no_grad():
            for i in range(len(data['metadata'])):
                # Fixed per-window seed makes reruns comparable regardless of earlier splits.
                torch.manual_seed(config.seed+i)
                latent,joints,aux=model(data['poses_input'][i:i+1].to(device),data['joints_input'][i:i+1].to(device),data['scenes'][i:i+1].to(device),data['gazes'][i:i+1].to(device),data['occ_map'][i:i+1].to(device),return_aux=True)
                predictions.append(joints.cpu());hiddens.append(aux['future_hidden'].cpu());latents.append(latent.cpu())
        prediction=torch.cat(predictions)
        result=dict(zip(METRICS,errors(prediction[:,6:],data['joints_label']).mean(0).tolist()))
        torch.save({'prediction':prediction,'future_hidden':torch.cat(hiddens),'trajectory_hidden':torch.cat(trajectory_hidden),'latent':torch.cat(latents),'checkpoint':args.checkpoint,'config':vars(config),'metadata':data['metadata']},output/(split+'.pt'))
        (output/(split+'.json')).write_text(json.dumps({'samples':len(prediction),'metrics_mm':result},indent=2)+'\n')
        print(split,result,flush=True)
    hook.remove()

if __name__=='__main__':main()
