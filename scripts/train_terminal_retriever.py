"""Train a full-bank retrieval objective with explicit terminal-pose supervision.

Queries contain observed inputs only. Candidate targets are training-bank futures;
validation is used solely for model selection, and test data is never loaded here.
"""
import argparse,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.multimodal_rag import MotionBank,retriever_from_bank,reference_frame
from scripts.rag_data import load_split


def full_bank_terminal_loss(similarity, distance, legal, temperature=.05):
    if similarity.shape != distance.shape or legal.shape != similarity.shape:
        raise ValueError('Scores, target distances and candidate masks must match')
    if not legal.any(-1).all():
        raise ValueError('Every training query requires a legal other-recording candidate')
    target=torch.softmax((-distance/temperature).masked_fill(~legal,-torch.inf),-1)
    log_pred=torch.log_softmax((similarity/.1).masked_fill(~legal,-torch.inf),-1).masked_fill(~legal,0.)
    return (target*(target.clamp_min(1e-12).log()-log_pred)).sum(-1).mean()


def terminal_fusion_loss(similarity, candidate_endpoints, legal):
    if not legal.any(-1).all():
        raise ValueError('Terminal fusion requires legal candidates for every query')
    weights=torch.softmax((similarity/.1).masked_fill(~legal,-torch.inf),-1)
    prior=torch.einsum('bn,njc->bjc',weights,candidate_endpoints)
    return (prior-candidate_endpoints).norm(dim=-1).mean()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--validation_data',required=True)
    p.add_argument('--baseline_cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=150);p.add_argument('--patience',type=int,default=25)
    p.add_argument('--endpoint_weight',type=float,default=10.);p.add_argument('--temperature',type=float,default=.05)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--scene_targets',action='store_true',help='Match targets to the legal same-scene candidate tier')
    p.add_argument('--allow_disjoint_recordings',action='store_true',help='Allow same-recording targets only when sampled intervals do not overlap')
    p.add_argument('--pose_loss_weight',type=float,default=0.,help='Train candidate fusion with supervised terminal-pose error')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    if a.endpoint_weight<0 or a.temperature<=0 or a.epochs<1 or a.pose_loss_weight<0: p.error('Invalid supervision parameters')
    torch.set_num_threads(4);torch.manual_seed(a.seed);device=torch.device(a.device)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    bank=MotionBank.load(a.bank);model=retriever_from_bank(bank,dropout=.1)
    def prepare(items):
        groups=[]
        with torch.no_grad():
            for start in range(0,len(items),8):
                group=items[start:start+8]
                groups.append(model.prepare_features(torch.stack([i.history_motion for i in group]),torch.stack([i.history_joints for i in group]),
                                                     torch.stack([i.history_gaze for i in group]),torch.stack([i.scene for i in group])))
        return tuple(torch.cat([g[i] for g in groups]).to(device) for i in range(3))
    features=prepare(bank.items)
    canonical=torch.stack([(i.future_joints-i.future_joints[:,:1])@reference_frame(i.history_joints)[1] for i in bank.items]).to(device)
    legal=torch.stack([bank.legal_candidate_mask(item.metadata(),exclude_same_sequence=not a.allow_disjoint_recordings)
                       for item in bank.items]).to(device)
    if a.scene_targets:
        scenes=[item.scene_id for item in bank.items]
        same_scene=torch.tensor([[left==right for right in scenes] for left in scenes],device=device,dtype=torch.bool)
        legal=torch.where((legal&same_scene).any(-1)[:,None],legal&same_scene,legal)
    distances=[]
    with torch.no_grad():
        for start in range(0,len(bank),16):
            errors=(canonical[start:start+16,None]-canonical[None]).norm(dim=-1)
            distances.append((errors.mean((-1,-2))+a.endpoint_weight*errors[:,:,-1].mean(-1))/(1+a.endpoint_weight))
    distance=torch.cat(distances)
    valid=load_split(a.validation_data,'valid')
    forecast=torch.load(Path(a.baseline_cache)/'valid.pt',map_location='cpu')
    if forecast['metadata']!=valid['metadata']:raise ValueError('Validation cache mismatch')
    valid_basis=torch.stack([reference_frame(h)[1] for h in valid['joints_input']])
    target=(valid['joints_label']-valid['joints_label'][:,:,:1])@valid_basis[:,None]
    base=forecast['prediction'][:,6:];base=(base-base[:,:,:1])@valid_basis[:,None]
    with torch.no_grad():
        valid_features=tuple(v.to(device) for v in model.prepare_features(valid['poses_input'],valid['joints_input'],valid['gazes'],valid['scenes']))
    model=model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    best=float('inf');best_epoch=-1;selected=None
    (out/'config.json').write_text(json.dumps(vars(a),indent=2)+'\n')
    for epoch in range(a.epochs):
        model.train();keys=model.encode_features(*features)
        similarity=keys@keys.T
        ranking_loss=full_bank_terminal_loss(similarity,distance,legal,a.temperature)
        loss=ranking_loss
        if a.pose_loss_weight:
            pose_loss=terminal_fusion_loss(similarity,canonical[:,-1],legal)
            loss=loss+a.pose_loss_weight*pose_loss
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval()
        with torch.no_grad():
            keys=model.encode_features(*features);bank.keys=keys.cpu();bank._keys_cache={}
            query=model.encode_features(*valid_features)
            result=bank.retrieve(query,5,valid['metadata'],unique_sequences=not a.scene_targets,
                                 prefer_same_scene=a.scene_targets,exclude_same_sequence=not a.allow_disjoint_recordings)
            if not result['candidate_valid_mask'].all():raise ValueError('Insufficient validation candidates')
            ids=result['retrieved_ids'].cpu();prior=canonical.cpu()[ids].mean(1)
            blended=.69266355*base+.30733645*prior
            error=(blended-target).norm(dim=-1)
            endpoint=float(error[:,-1].mean()*1000);mean=float(error.mean()*1000)
            record={'epoch':epoch,'loss':float(loss),'ranking_loss':float(ranking_loss),
                    'terminal_pose_loss_m':float(pose_loss) if a.pose_loss_weight else 0.,
                    'valid_mean_mpjpe_mm':mean,'valid_endpoint_mpjpe_mm':endpoint}
            score=endpoint+.1*mean
            if score<best:
                best=score;best_epoch=epoch;selected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save({'state_dict':selected,'encoder_version':model.encoder_version,'embedding_dim':model.embedding_dim},out/'best.pt')
                (out/'best_metrics.json').write_text(json.dumps(record,indent=2)+'\n')
        with (out/'metrics.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if epoch-best_epoch>=a.patience:break
    model.load_state_dict(selected,strict=True);bank.encode_keys(model,device=device,batch_size=8)
    bank.manifest.update(retriever_trained=True,retriever_objective='full-bank terminal-weighted distance KL',
                         endpoint_weight=a.endpoint_weight,selection_split='valid',selected_epoch=best_epoch,
                         scene_target_supervision=a.scene_targets,disjoint_recording_targets=a.allow_disjoint_recordings,
                         terminal_pose_loss_weight=a.pose_loss_weight)
    bank.save(out/'trained.bank.pt')
    (out/'summary.json').write_text(json.dumps({'epochs_completed':epoch+1,'best_epoch':best_epoch,'best_valid_score':best},indent=2)+'\n')

if __name__=='__main__':main()
