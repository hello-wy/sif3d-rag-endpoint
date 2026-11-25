import numpy as np
import torch
import trimesh

from scripts.expand_endpoint_training_data import read_gaze_point


def test_gaze_point_cloud_without_faces_is_read(tmp_path):
    folder = tmp_path / 'eye_pc'
    folder.mkdir()
    trimesh.PointCloud([[2., 4., 6.]]).export(folder / '15_center.ply')
    trimesh.PointCloud([[2., 4., 6.], [3., 5., 7.]]).export(folder / '15.ply')
    point = read_gaze_point(tmp_path, 15, 2., np.eye(4))
    assert torch.allclose(point, torch.tensor([1., 2., 3.]))
    assert read_gaze_point(tmp_path, 30, 2., np.eye(4)) is None
