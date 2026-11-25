import torch
import json
from pathlib import Path
import time
import smplx
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import gimo_dataset
from human_body_prior.tools.model_loader import load_vposer
from config.config import MotionFromGazeConfig
from model.sif3d_mod import SIF3D
from utils.logger import MetricTracker
import csv

class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        self.test_loader = DataLoader(
            gimo_dataset.EgoEvalDataset(config, train=False),
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vposer, _ = load_vposer(self.config.vposer_path, vp_model='snapshot')
        self.vposer = self.vposer.to(self.device)

        self.body_model = smplx.create(self.config.smplx_path, model_type='smplx', gender='neutral', ext='npz',
                                       num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                       create_betas=True, create_left_hand_pose=True, create_right_hand_pose=True,
                                       create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                       create_reye_pose=True, create_transl=True, num_betas=10, num_expression_coeffs=10,
                                       batch_size=config.output_seq_len + config.input_seq_len,
                                       ).to(self.device)

    def eval(self):
        torch.manual_seed(self.config.seed)
        model = SIF3D(self.config, self.vposer, self.body_model).to(self.device)
        model = model.to(self.device)
        assert self.config.load_model_dir is not None
        print('loading pretrained model from ', self.config.load_model_dir)
        state_dict = torch.load(self.config.load_model_dir, map_location='cpu')
        model.load_prediction_checkpoint(state_dict)
        print('load done!')

        with torch.no_grad():
            model.eval()
            test_metrics = MetricTracker('loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe')
            pre_refiner_metrics = []
            sample_records = []
            joint_records, pre_refiner_records = [], []
            pelvis_change_mm = 0.0
            for i, data in enumerate(self.test_loader):
                torch.manual_seed(self.config.seed+i)
                gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene, occ = data

                gazes = gazes.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)
                occ = occ.to(self.device)

                rag_metadata = None
                if getattr(self.config, 'use_rag', False):
                    rag_metadata = [
                        {'sequence_id': str(s), 'scene_id': str(sc),
                         'sample_id': f'{s}:{int(self.test_loader.dataset.poses_path_list[i])}'}
                        for s, sc in zip(seq, scene)
                    ]
                model_output = model(
                    poses_input, joints_input[:, :, :23], scene_points, gazes, occ,
                    rag_metadata=rag_metadata,
                    return_aux=bool(getattr(self.config, 'use_rag', False) or getattr(self.config, 'use_refiner', False)),
                )
                if len(model_output) == 3:
                    poses_predict, joints_predict, aux = model_output
                    pre = self.calc_loss_gcn(aux['pre_refiner_joints'], joints_label[:, :, :23], joints_input[:, :, :23])
                    pre_refiner_metrics.append((float(pre[0][:, 6:].mean()), float(pre[2][:, 6:].mean())))
                    before = aux['pre_refiner_joints']
                else:
                    poses_predict, joints_predict = model_output
                    before = joints_predict
                root_difference = (joints_predict[:, self.config.input_seq_len:, 0] - before[:, self.config.input_seq_len:, 0]).abs()
                pelvis_change_mm = max(pelvis_change_mm, float(root_difference.max()) * 1000)
                if root_difference.any():
                    raise ValueError('RAG refiner changed the frozen pelvis trajectory')
                joint_records.append(joints_predict.cpu())
                pre_refiner_records.append(before.cpu())
                # from thop import profile
                # total_flops, total_params = profile(model, inputs=(poses_input, joints_input[:, :, :23], scene_points, gazes))
                # print(f'FLOPs: {total_flops / 1e9:.2f}G, Params: {total_params / 1e6:.2f}M')
                
                loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = \
                    self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])

                test_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), gazes.shape[0])
                test_metrics.update("loss_des_trans", loss_des_trans_gcn, gazes.shape[0])
                test_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), gazes.shape[0])
                test_metrics.update("des_mpjpe", des_mpjpe_gcn, gazes.shape[0])
                
                output = Path(self.config.output_path);output.mkdir(parents=True, exist_ok=True)
                sample_records.append({'sequence': seq[0], 'scene': scene[0], 'translation_mm': float(loss_trans_gcn[:, 6:].mean())*1000,
                                       'endpoint_translation_mm': float(loss_des_trans_gcn)*1000, 'mpjpe_mm': float(mpjpe_gcn[:, 6:].mean())*1000,
                                       'endpoint_mpjpe_mm': float(des_mpjpe_gcn)*1000})
                with open(output / 'raw_pred_seq.csv', 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    row = [
                        seq[0],
                        scene[0],
                        loss_trans_gcn[:, 6:].cpu().numpy().mean(),
                        loss_des_trans_gcn.cpu().numpy(),
                        mpjpe_gcn[:, 6:].cpu().numpy().mean(),
                        des_mpjpe_gcn.cpu().numpy()
                    ]
                    if len(model_output) == 3:
                        row.extend([pre_refiner_metrics[-1][0], pre_refiner_metrics[-1][1]])
                    writer.writerow(row)

            print(test_metrics.result())
            (output / 'metrics.json').write_text(json.dumps({'samples': len(sample_records), 'metrics_mm': {k: v*1000 for k,v in test_metrics.result().items()}, 'per_sample': sample_records,
                                                           'pelvis_max_coordinate_change_mm': pelvis_change_mm, 'config': vars(self.config)}, indent=2)+'\n')
            torch.save({'prediction': torch.cat(joint_records), 'pre_refiner': torch.cat(pre_refiner_records),
                        'sample_ids': [f"{row['sequence']}:{int(self.test_loader.dataset.poses_path_list[i])}" for i,row in enumerate(sample_records)]},
                       output / 'predictions.pt')
            if pre_refiner_metrics:
                trans = sum(x[0] for x in pre_refiner_metrics) / len(pre_refiner_metrics)
                mpjpe = sum(x[1] for x in pre_refiner_metrics) / len(pre_refiner_metrics)
                print({'pre_refiner_trans': trans, 'pre_refiner_mpjpe': mpjpe})
            test_metrics.reset()



    def calculate_loss(self, poses_predict, poses_label, poses_input, joints_input, joints_label, test=False):
        poses_label = torch.cat([poses_input, poses_label], dim=1)

        loss_des_ori = F.l1_loss(poses_predict[:, -1, :3], poses_label[:, -1, :3])
        loss_des_trans = torch.norm(poses_predict[:, :, 3:6] - poses_label[:, :, 3:6], dim=-1)[:, -1].mean()
        loss_des_latent = F.l1_loss(poses_predict[:, -1, 6:], poses_label[:, -1, 6:])

        loss_all = F.l1_loss(poses_predict[:, :, :], poses_label[:, :, :], reduction='none')

        loss_ori = loss_all[:, :, :3]
        loss_trans = torch.norm(poses_predict[:, :, 3:6] - poses_label[:, :, 3:6], dim=-1)
        loss_latent = loss_all[:, :, 6:]

        pred_pose = self.vposer.decode(poses_predict[:, :, 6:], output_type='aa').view(-1, 63)
        pred_smplx = self.body_model(return_verts=True, body_pose=pred_pose,
                                     global_orient=poses_predict[:, :, :3].view(-1, 3),
                                     transl=poses_predict[:, :, 3:6].view(-1, 3),
                                     pose_embedding=poses_predict[:, :, 6:].view(-1, 32)).joints

        gt_smplx = torch.cat([joints_input, joints_label], dim=1)
        pred_smplx = pred_smplx.view(-1, self.config.input_seq_len + self.config.output_seq_len, 127, 3)[:, :, :23]
        gt_smplx -= gt_smplx[:, :, [0]]
        pred_smplx -= pred_smplx[:, :, [0]]

        mpjpe = torch.norm(gt_smplx - pred_smplx, dim=-1)

        return loss_ori, loss_trans, loss_latent, loss_des_ori, loss_des_trans, loss_des_latent, mpjpe, mpjpe[:, -1].mean()

    def calc_loss_gcn(self, poses_predict, poses_label, poses_input):
        poses_label = torch.cat([poses_input, poses_label], dim=1)

        loss_trans = torch.norm(poses_predict[:, :, 0] - poses_label[:, :, 0], dim=-1)

        poses_label = poses_label - poses_label[:, :, [0]]
        poses_predict = poses_predict - poses_predict[:, :, [0]]
        mpjpe = torch.norm(poses_predict - poses_label, dim=-1)             # bs * seq * J

        return loss_trans, loss_trans[:, -1].mean(), mpjpe, mpjpe[:, -1].mean()


if __name__ == '__main__':
    config = MotionFromGazeConfig().parse_args()
    start = time.time()
    evaluator = SMPLX_evalutor(config)
    evaluator.eval()
