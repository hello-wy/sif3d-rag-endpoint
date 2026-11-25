"""Full-split matched evaluation of a frozen predictor cache and RAG checkpoint."""
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.multimodal_rag import MotionBank,retriever_from_bank,PrototypeEncoder,MotionRefiner,LocalMotionRefiner,PrototypeBlendRefiner,EndpointPrototypeRefiner,TerminalBlendRefiner,AdaptiveTerminalRefiner,forecast_candidate_scores
from scripts.rag_data import load_split,errors,METRICS
from scripts.train_rag_refiner import prepare_features


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--split',choices=['valid','test'],default='test')
    p.add_argument('--random_seed',type=int);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    root=Path(a.run);saved=json.loads((root/'config.json').read_text());c=saved['config']
    checkpoint=torch.load(root/'best.pth',map_location='cpu');bank=MotionBank.load(c['rag_bank'])
    encoder=retriever_from_bank(bank).eval();data=load_split(saved['training']['data'],a.split)
    cache=torch.load(Path(saved['baseline_cache'])/(a.split+'.pt'),map_location='cpu')
    if cache['metadata']!=data['metadata']:raise ValueError('Metadata mismatch')
    local=c['rag_refiner_type']=='local'
    features=prepare_features(data,cache,bank,encoder,torch.device('cpu'),local,c.get('rag_forecast_rerank',False),c['rag_refiner_type']=='endpoint',c.get('rag_unique_sequences',False),
                              c.get('rag_prefer_same_scene',False),c.get('rag_exclude_same_sequence',True),c['rag_top_k'])
    if a.random_seed is not None:
        torch.manual_seed(a.random_seed)
        k=c['rag_top_k'];ids=torch.full((len(data['metadata']),k),-1,dtype=torch.long)
        for b,meta in enumerate(data['metadata']):
            legal=torch.where(bank.legal_candidate_mask(meta,c.get('rag_exclude_same_sequence',True),c.get('rag_exclude_scene',False)))[0].tolist()
            order=torch.randperm(len(legal)).tolist()
            if c.get('rag_prefer_same_scene',False):
                order.sort(key=lambda pos:bank.items[legal[pos]].scene_id!=meta['scene_id'])
            if c.get('rag_unique_sequences',False):
                chosen=[];seen=set()
                for position in order:
                    index=legal[position];sequence=bank.items[index].sequence_id
                    if sequence in seen:continue
                    seen.add(sequence);chosen.append(index)
                    if len(chosen)==k:break
                ids[b,:len(chosen)]=torch.tensor(chosen)
            else:
                take=min(len(legal),k);ids[b,:take]=torch.tensor([legal[pos] for pos in order[:take]])
        future=bank.gather_future(ids,data['joints_input'])
        random_scores = forecast_candidate_scores(cache['prediction'][:,6:],future,ids>=0,local) if c.get('rag_forecast_rerank',False) else torch.where(ids>=0,0.,-torch.inf)
        if local:
            from model.multimodal_rag import reference_frame,to_reference
            refs=[reference_frame(row) for row in data['joints_input']]
            future=torch.stack([to_reference(row,o,basis) for row,(o,basis) in zip(future,refs)])
        features.update(future=future,valid=ids>=0,scores=random_scores)
        if c['rag_refiner_type']=='endpoint':
            legal=features['endpoint_valid']
            random_mask=torch.zeros_like(legal)
            for b in range(len(legal)):
                candidates=torch.where(legal[b])[0]
                chosen=candidates[torch.randperm(len(candidates))[:c['rag_endpoint_topk']]]
                random_mask[b,chosen]=True
            features['endpoint_valid']=random_mask
    prototype=PrototypeEncoder(23,c['rag_embedding_dim'])
    prototype.load_state_dict({k.removeprefix('prototype_encoder.'):v for k,v in checkpoint.items() if k.startswith('prototype_encoder.')},strict=True)
    klass={'local':LocalMotionRefiner,'legacy':MotionRefiner,'blend':PrototypeBlendRefiner,'endpoint':EndpointPrototypeRefiner,'terminal_blend':TerminalBlendRefiner,'adaptive_terminal':AdaptiveTerminalRefiner}[c['rag_refiner_type']]
    refiner=klass(context_dim=c['rag_embedding_dim'],prototype_dim=c['rag_embedding_dim'],hidden_dim=c['rag_refiner_hidden_dim'],
                  **({'max_delta':c['rag_max_delta']} if local or c['rag_refiner_type']=='adaptive_terminal' else {}),
                  **({'endpoint_topk':c['rag_endpoint_topk'],'endpoint_frames':c['rag_endpoint_frames']} if c['rag_refiner_type']=='endpoint' else {'endpoint_frames':c['rag_endpoint_frames']} if c['rag_refiner_type']=='terminal_blend' else {'endpoint_frames':c['rag_endpoint_frames'],'joint_trust':c['rag_terminal_joint_trust'],'part_trust':c.get('rag_terminal_part_trust',False),'max_trust':c.get('rag_terminal_max_trust',.8)} if c['rag_refiner_type']=='adaptive_terminal' else {}))
    refiner.load_state_dict({k.removeprefix('motion_refiner.'):v for k,v in checkpoint.items() if k.startswith('motion_refiner.')},strict=True)
    prototype.eval();refiner.eval()
    predictions=[]
    with torch.no_grad():
        for i in range(len(data['metadata'])):
            pred,_,_=refiner(features['motion'][i:i+1],features['base'][i:i+1],prototype(features['future'][i:i+1]),
                             features['query'][i:i+1],features['scores'][i:i+1],features['valid'][i:i+1],
                             **({'history_joints':features['history'][i:i+1]} if local or c['rag_refiner_type']=='adaptive_terminal' else {}),
                             **({'prototype_future':features['future'][i:i+1]} if c['rag_refiner_type'] in ('blend','endpoint','terminal_blend','adaptive_terminal') else {}),
                             **({k:features[k][i:i+1] for k in ('endpoint_future','endpoint_valid')} if c['rag_refiner_type']=='endpoint' else {}))
            predictions.append(pred)
    prediction=torch.cat(predictions);values=errors(prediction,features['target']);baseline=errors(features['base'],features['target'])
    pelvis_delta_mm = float((prediction[:,:,0]-features['base'][:,:,0]).abs().max()*1000)
    if pelvis_delta_mm != 0:raise ValueError('Trajectory changed')
    report={'pelvis_max_coordinate_change_mm':pelvis_delta_mm,'samples':len(prediction),'split':a.split,'random_seed':a.random_seed,'metrics_mm':dict(zip(METRICS,values.mean(0).tolist())),
            'baseline_metrics_mm':dict(zip(METRICS,baseline.mean(0).tolist())),
            'per_sample':[{'sample_id':m['sample_id'],**dict(zip(METRICS,row.tolist()))} for m,row in zip(data['metadata'],values)]}
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n');print(report['metrics_mm'])
    torch.save({'prediction': prediction, 'baseline': features['base'],
                'sample_ids': [m['sample_id'] for m in data['metadata']]}, output.with_suffix('.pt'))

if __name__=='__main__':main()
