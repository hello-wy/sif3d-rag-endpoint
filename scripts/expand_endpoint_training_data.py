"""Build time-jittered training windows from raw poses, with test-frame and recording holdout guards."""
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch,smplx,trimesh
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from human_body_prior.tools.model_loader import load_vposer
from scripts.rag_data import load_split


def read_gaze_point(directory, frame, scale, norm):
    center = directory / 'eye_pc' / f'{frame}_center.ply'
    cloud = directory / 'eye_pc' / f'{frame}.ply'
    if not center.is_file() or not cloud.is_file():
        return None
    # GIMO gaze PLYs are point clouds without faces. load_mesh discards their vertices.
    center_points = np.asarray(trimesh.load(center, force='point').vertices)
    cloud_points = np.asarray(trimesh.load(cloud, force='point').vertices)
    if not len(center_points) or not len(cloud_points) or not np.isfinite(center_points[0]).all():
        return None
    if np.abs(center_points[0]).sum() <= 1e-8 or np.abs(cloud_points).sum() <= 1e-8:
        return None
    point = norm[:3, :3] @ (center_points[0] / scale) + norm[:3, 3]
    return torch.from_numpy(point).float()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--dataroot',default='/data/wuyang/human/GIMO');p.add_argument('--output',required=True);p.add_argument('--offsets',type=int,nargs='+',default=[-30,-15,0,15,30])
    a=p.parse_args();torch.set_num_threads(4);source=Path(a.source);raw=Path(a.dataroot);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    original=load_split(source,'train');test_rows=pd.read_csv(source/'test.csv');heldout=set(pd.read_csv(source/'valid.csv').sequence_path)
    vp,_=load_vposer('/home/customer/sif3d-raw/vposer_v1_0',vp_model='snapshot');vp=vp.cuda().eval();body=smplx.create('/home/customer/sif3d-raw/data',model_type='smplx',gender='neutral',ext='npz',num_pca_comps=12,batch_size=16).cuda().eval()
    fields={k:[] for k in ['gazes','poses_input','poses_label','joints_input','joints_label','occ_map','scene_points_32768']};rows=[];seen=set();failed=[]
    for index,row in original['rows'].iterrows():
        sequence=str(row.sequence_path);scene=str(row.scene);directory=raw/scene/sequence
        if sequence in heldout:raise ValueError('Validation recording in train')
        transform=json.loads((directory/row.transformation).read_text());scale=transform['scale'];pose_transform=np.asarray(transform['transformation'],dtype=np.float64);pose_transform[:3,3]/=scale
        norm=np.loadtxt(raw/scene/'scene_obj/transform_norm.txt').reshape(4,4);norm[:3,3]/=scale;full=norm@pose_transform
        for offset in a.offsets:
            start=int(row.start_frame)+offset;sample_id=f'{sequence}:{start}'
            if start<0 or sample_id in seen:continue
            if any(str(t.sequence_path)==sequence and max(start,int(t.start_frame))<=min(start+225,int(t.start_frame)+225) for _,t in test_rows.iterrows()):continue
            frames=[start+15*j for j in range(16)]
            if any(not (directory/'smplx_local'/f'{f}.pkl').is_file() for f in frames):continue
            try:
                motion=[];gaze=[]
                for j,frame in enumerate(frames):
                    with (directory/'smplx_local'/f'{frame}.pkl').open('rb') as handle:data=pickle.load(handle)
                    ori=data['orient'].detach().cpu().numpy();trans=data['trans'].detach().cpu().numpy().reshape(3)
                    orientation=Rotation.from_matrix(full[:3,:3]@Rotation.from_rotvec(ori).as_matrix()).as_rotvec();position=full[:3,:3]@trans+full[:3,3]
                    motion.append(torch.cat([torch.from_numpy(orientation).float(),torch.from_numpy(position).float(),data['latent'].detach().cpu().reshape(-1).float()]))
                    if j<6:
                        point = read_gaze_point(directory, frame, scale, norm)
                        gaze.append(point if point is not None else torch.zeros(3))
                motion=torch.stack(motion);gaze=torch.stack(gaze)[:,None];valid=gaze.norm(dim=-1).squeeze(-1)>1e-6
                gaze[~valid]=gaze[valid].mean(0) if valid.any() else original['gazes'][index].mean(0)
                if offset==0:
                    expected=torch.cat([original['poses_input'][index],original['poses_label'][index]],0)
                    if not torch.allclose(motion,expected,atol=1e-4):raise ValueError('Raw/cache motion mismatch at original start')
                    joints=torch.cat([original['joints_input'][index],original['joints_label'][index]],0);gaze=original['gazes'][index]
                else:
                    with torch.no_grad():
                        m=motion.cuda();pose=vp.decode(m[:,6:],output_type='aa').reshape(16,63);j=body(body_pose=pose,global_orient=m[:,:3],transl=m[:,3:6],pose_embedding=m[:,6:]).joints[:,:23];joints=j.cpu()
                record=row.copy();record.start_frame=start;record.end_frame=max(int(row.end_frame),start+225);rows.append(record);seen.add(sample_id)
                for name,value in [('gazes',gaze),('poses_input',motion[:6]),('poses_label',motion[6:]),('joints_input',joints[:6]),('joints_label',joints[6:]),('occ_map',original['occ_map'][index]),('scene_points_32768',original['scenes'][index])]:fields[name].append(value.clone())
            except (OSError,ValueError,RuntimeError) as error:
                failed.append({'sample_id':sample_id,'error':str(error)})
                if offset==0:raise
        if index%20==0:print('source',index,'windows',len(rows),flush=True)
    target=out/'cache/train';target.mkdir(parents=True)
    for name,values in fields.items():torch.save(torch.stack(values),target/(name+'.pth'))
    pd.DataFrame(rows).to_csv(out/'train.csv',index=False)
    for split in ['valid','test']:
        import shutil
        shutil.copy2(source/(split+'.csv'),out/(split+'.csv'));shutil.copytree(source/'cache'/split,out/'cache'/split)
    manifest={'source':str(source),'offsets':a.offsets,'train_samples':len(rows),'heldout_recordings':sorted(heldout),'rejected':failed,'test_frame_overlap_guard':True,'note':'New raw-frame supervision from time-jittered training intervals; action labels are not used.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('finished',len(rows),flush=True)

if __name__=='__main__':main()
