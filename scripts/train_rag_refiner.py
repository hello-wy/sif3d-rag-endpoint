"""Train prototype-conditioned refinement against fixed SIF3D forecasts; select on held-out recordings."""
import argparse,hashlib,json,random,sys
from pathlib import Path
import torch,smplx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config.config import MotionFromGazeConfig
from model.sif3d_mod import SIF3D
from model.multimodal_rag import MotionBank,retriever_from_bank,reference_frame,to_reference,forecast_candidate_scores,endpoint_candidates
from human_body_prior.tools.model_loader import load_vposer
from scripts.rag_data import load_split,errors,METRICS


def prepare_features(data,baseline,bank,encoder,device,local,rerank=False,endpoint=False,unique_sequences=False,
                     prefer_same_scene=False,exclude_same_sequence=True,top_k=5):
    with torch.no_grad():
        queries=[]
        for start in range(0,len(data['metadata']),16):
            end=start+16
            queries.append(encoder(data['poses_input'][start:end].to(device),data['joints_input'][start:end].to(device),
                                   data['gazes'][start:end].to(device),data['scenes'][start:end].to(device)))
        query=torch.cat(queries)
        result=bank.retrieve(query,top_k=top_k,metadata=data['metadata'],unique_sequences=unique_sequences,
                             prefer_same_scene=prefer_same_scene,exclude_same_sequence=exclude_same_sequence)
        future=bank.gather_future(result['retrieved_ids'].cpu(),data['joints_input'])
        if rerank:
            result['retrieval_scores'] = forecast_candidate_scores(baseline['prediction'][:,6:].to(device),
                                                                  future.to(device), result['candidate_valid_mask'], local)
        if local:
            refs=[reference_frame(row) for row in data['joints_input']]
            future=torch.stack([to_reference(row,o,b) for row,(o,b) in zip(future,refs)])
    endpoint_features = {}
    if endpoint:
        with torch.no_grad():
            endpoint_future, endpoint_valid = endpoint_candidates(bank, data['joints_input'], data['metadata'])
        endpoint_features = {'endpoint_future': endpoint_future.to(device), 'endpoint_valid': endpoint_valid.to(device)}
    return {**endpoint_features, 'motion':data['poses_input'].to(device),'history':data['joints_input'].to(device),
            'target':data['joints_label'].to(device),'base':baseline['prediction'][:,6:].to(device),
            'future':future.to(device),'query':query,'scores':result['retrieval_scores'],'valid':result['candidate_valid_mask']}


def forward(model,features,ids,local):
    f={k:v[ids] for k,v in features.items()}
    tokens=model.prototype_encoder(f['future'])
    pred,_,delta=model.motion_refiner(f['motion'],f['base'],tokens,f['query'],f['scores'],f['valid'],
                                    **({'history_joints':f['history']} if local or model.config.rag_refiner_type=='adaptive_terminal' else {}),
                                    **({'prototype_future':f['future']} if model.config.rag_refiner_type in ('blend','endpoint','terminal_blend','adaptive_terminal') else {}),
                                    **({k:f[k] for k in ('endpoint_future','endpoint_valid')} if model.config.rag_refiner_type=='endpoint' else {}))
    return pred,delta


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--baseline_cache',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--patience',type=int,default=15)
    p.add_argument('--regularization',type=float,default=0.1)
    p.add_argument('--endpoint_loss_weight',type=float,default=0.5)
    p.add_argument('--selection',choices=['mean','endpoint'],default='mean')
    p.add_argument('--refiner_init',default=None,help='Initialize only new refiner weights from a verified matching-bank RAG checkpoint')
    args,remaining=p.parse_known_args();config=MotionFromGazeConfig().parse_args(remaining)
    config.use_rag=True;config.rag_train_stage='refiner';config.use_refiner=True
    torch.set_num_threads(4);torch.manual_seed(config.seed);random.seed(config.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    vposer,_=load_vposer(config.vposer_path,vp_model='snapshot')
    body=smplx.create(config.smplx_path,model_type='smplx',gender='neutral',ext='npz',num_pca_comps=12,batch_size=16).to(device)
    model=SIF3D(config,vposer.to(device),body).to(device)
    weights=torch.load(args.checkpoint,map_location='cpu');model.load_prediction_checkpoint(weights,allow_baseline=True)
    if args.refiner_init:
        initialization=torch.load(args.refiner_init,map_location='cpu')
        if not torch.equal(initialization.get('rag_bank_signature',torch.empty(0,dtype=torch.uint8)),model.rag_bank_signature.cpu()):
            raise ValueError('Refiner initialization uses a different bank')
        for key,value in weights.items():
            if not key.startswith('smplx_model.') and key in initialization and not torch.equal(value.cpu(),initialization[key].cpu()):
                raise ValueError(f'Refiner initialization backbone mismatch: {key}')
        refiner_state={k.removeprefix('motion_refiner.'):v for k,v in initialization.items() if k.startswith('motion_refiner.')}
        if hasattr(model.motion_refiner,'trust_limit'):
            refiner_state['trust_limit']=model.motion_refiner.trust_limit.cpu().clone()
        model.motion_refiner.load_state_dict(refiner_state,strict=True)
    features={}
    for split in ['train','valid']:
        data=load_split(args.data,split)
        baseline=torch.load(Path(args.baseline_cache)/(split+'.pt'),map_location='cpu')
        if baseline['metadata']!=data['metadata']:raise ValueError('Forecast cache metadata mismatch')
        features[split]=prepare_features(data,baseline,model.rag_bank,model.rag_retriever,device,
                                        config.rag_refiner_type=='local',config.rag_forecast_rerank,config.rag_refiner_type=='endpoint',config.rag_unique_sequences,
                                        config.rag_prefer_same_scene,config.rag_exclude_same_sequence,config.rag_top_k)
    if config.rag_refiner_type=='adaptive_terminal':
        with torch.no_grad():
            f=features['train'];obs=model.motion_refiner.observation_features(f['base'],f['future'],f['valid'],f['history'])
            model.motion_refiner.feature_mean.copy_(obs.mean(0));model.motion_refiner.feature_std.copy_(obs.std(0).clamp_min(.05))
    config_data={'config':vars(config),'training':vars(args),'selection_split':'valid recordings',
                 'backbone_checkpoint':args.checkpoint,'baseline_cache':args.baseline_cache}
    (output/'config.json').write_text(json.dumps(config_data,indent=2)+'\n')
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=config.lr,weight_decay=1e-3)
    best=float('inf');best_epoch=-1;best_modules=None;local=config.rag_refiner_type=='local'
    for epoch in range(args.epochs):
        model.train();ids=torch.randperm(len(features['train']['base']),device=device);losses=[]
        for start in range(0,len(ids),config.batch_size):
            batch=ids[start:start+config.batch_size];target=features['train']['target'][batch]
            prediction,delta=forward(model,features['train'],batch,local)
            relative=prediction-prediction[:,:,:1];gt=target-target[:,:,:1]
            e=(relative-gt).norm(dim=-1)
            velocity=(torch.diff(relative,dim=1)-torch.diff(gt,dim=1)).abs().mean()
            loss=e.mean()+args.endpoint_loss_weight*e[:,-1].mean()+0.1*velocity+args.regularization*delta.square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.0);optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            prediction,_=forward(model,features['valid'],torch.arange(len(features['valid']['base']),device=device),local)
            validation=errors(prediction,features['valid']['target']).mean(0).tolist()
        result={'epoch':epoch,'train_loss':sum(losses)/len(losses),**dict(zip(METRICS,validation))}
        with (output/'metrics.jsonl').open('a') as handle:handle.write(json.dumps(result)+'\n')
        print(json.dumps(result),flush=True)
        score = result['mpjpe_mm'] if args.selection=='mean' else result['endpoint_mpjpe_mm']+0.1*result['mpjpe_mm']
        if score<best:
            best=score;best_epoch=epoch
            # The backbone is frozen: avoid writing its large checkpoint on every
            # tiny validation improvement; persist just the selected RAG modules.
            best_modules={k:v.detach().cpu().clone() for k,v in model.state_dict().items()
                          if k.startswith(('prototype_encoder.','prototype_adapter.','motion_refiner.'))}
            torch.save(best_modules,output/'best_rag_modules.pt')
            (output/'best_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        if epoch-best_epoch>=args.patience:break
    torch.save({k:v.cpu() for k,v in model.state_dict().items() if not k.startswith('smplx_model.')},output/'last.pth')
    selected={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('smplx_model.')}
    selected.update(best_modules)
    torch.save(selected,output/'best.pth')
    frozen=[key for key,value in weights.items() if key in model.state_dict() and not key.startswith('smplx_model.')]
    changed=[key for key in frozen if not torch.equal(weights[key].cpu(),model.state_dict()[key].cpu())]
    if changed:raise ValueError(f'Frozen backbone changed: {changed}')
    (output/'training_summary.json').write_text(json.dumps({'epochs_completed':epoch+1,'best_epoch':best_epoch,'best_valid_selection_score':best,'selection':args.selection,'best_valid_metrics':json.loads((output/'best_metrics.json').read_text()),'changed_backbone_tensors':len(changed),'checked_backbone_tensors':len(frozen)},indent=2)+'\n')

if __name__=='__main__':main()
