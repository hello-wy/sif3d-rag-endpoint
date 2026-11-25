import torch
import pandas as pd

from model.endpoint_pose_bank import EndpointPoseBank, EndpointPoseItem, EndpointPoseRefiner, EndpointPoseTemporalRefiner, EndpointTrustRefiner, EndpointJointTrustRefiner, EndpointInterpolationRefiner, EndpointTerminalRefiner, EndpointLocalMotionRefiner
from model.multimodal_rag import MotionBank, MultiModalRetriever, reference_frame


def test_gimo_builder_keeps_only_last_label_pose():
    history = torch.randn(1, 6, 23, 3)
    labels = torch.randn(1, 10, 23, 3)
    motions = torch.randn(1, 10, 38)
    data = {'rows': pd.DataFrame([{'sequence_path': 'recording', 'scene': 'scene', 'start_frame': 0}]),
            'joints_input': history, 'joints_label': labels, 'poses_label': motions,
            'scenes': torch.randn(1, 20, 3)}
    bank = EndpointPoseBank.from_gimo_data(data)
    item = bank.items[0]
    basis = reference_frame(history[0])[1]
    assert torch.equal(item.pose_motion, motions[0, -1])
    assert torch.allclose(item.pose_joints, (labels[0, -1] - labels[0, -1, :1]) @ basis)
    assert not hasattr(item, 'future_joints')


def test_single_frame_bank_and_two_modality_query(tmp_path):
    history = torch.randn(2, 6, 23, 3)
    item = EndpointPoseItem('sample:0', 'GIMO', 'train', 'recording', 'scene',
                            tuple(range(6)), tuple(range(6, 16)),
                            torch.randn(38), torch.randn(23, 3))
    bank = EndpointPoseBank([item])
    path = tmp_path / 'pose.bank.pt'
    bank.save(path)
    loaded = MotionBank.load(path)
    assert isinstance(loaded, EndpointPoseBank)
    assert not hasattr(loaded.items[0], 'future_joints')
    assert not hasattr(loaded.items[0], 'history_motion')
    pose = loaded.gather_pose(torch.tensor([[0, -1], [-1, 0]]), history)
    for b, k in ((0, 0), (1, 1)):
        origin, basis = reference_frame(history[b])
        assert torch.allclose(pose[b, k], item.pose_joints @ basis.T + origin)
    assert pose[0, 1].eq(0).all()

    retriever = MultiModalRetriever(use_motion=False, scene_focus=True, dropout=0).eval()
    gaze = torch.randn(2, 6, 3)
    scene = torch.randn(2, 20, 3)
    first = retriever(None, history, gaze, scene)
    second = retriever(torch.randn(2, 6, 38) * 1000, history, gaze, scene)
    assert torch.equal(first, second)
    assert torch.equal(first, retriever(None, torch.randn_like(history) * 1000, gaze, scene))
    assert torch.equal(first, retriever(None, None, gaze, scene))
    assert retriever.encoder_version == 'multimodal-retriever-v5'
    assert retriever.prepare_features(None, history, gaze, scene)[2].shape == (2, 12)
    histogram_encoder = MultiModalRetriever(use_motion=False, scene_focus=True, scene_hist=True, dropout=0).eval()
    assert histogram_encoder.prepare_features(None, None, gaze, scene)[2].shape == (2, 60)
    assert torch.equal(histogram_encoder(None, history, gaze, scene),
                       histogram_encoder(None, None, gaze, scene))


def test_endpoint_refiner_preserves_pelvis_and_ignores_padding():
    for model in (EndpointPoseRefiner(), EndpointPoseTemporalRefiner().eval()):
        base = torch.randn(2, 10, 23, 3)
        poses = torch.randn(2, 3, 23, 3)
        valid = torch.tensor([[True, False, False], [False, False, False]])
        prediction, delta = model(base, poses, valid)
        assert torch.equal(prediction[:, :, 0], base[:, :, 0])
        assert torch.equal(prediction[1], base[1])
        poses[:, 1:] += 1000
        repeated, _ = model(base, poses, valid)
        assert torch.equal(prediction, repeated)
        prediction.square().mean().backward()
        assert model.trust_logits.grad is not None

    reranked = EndpointPoseRefiner(rerank_topk=1).eval()
    base = torch.zeros(1, 10, 23, 3)
    poses = torch.zeros(1, 2, 23, 3)
    poses[:, 0, 1:] = 10
    poses[:, 1, 1:] = .1
    selected, _ = reranked(base, poses, torch.ones(1, 2, dtype=torch.bool))
    assert selected.abs().max() < 1


def test_endpoint_trust_refiner_uses_candidates_without_moving_pelvis():
    for model in (EndpointTrustRefiner().eval(), EndpointJointTrustRefiner().eval()):
        base = torch.randn(2, 10, 23, 3)
        poses = torch.randn(2, 3, 23, 3)
        history = torch.randn(2, 6, 23, 3)
        valid = torch.tensor([[True, True, False], [False, False, False]])
        prediction, delta = model(base, poses, valid, history)
        assert torch.equal(prediction[:, :, 0], base[:, :, 0])
        assert torch.equal(prediction[1], base[1])
        poses[:, 2] += 1000
        repeated, _ = model(base, poses, valid, history)
        assert torch.equal(prediction, repeated)
        prediction.square().mean().backward()
        assert model.trust[-1].weight.grad is not None


def test_endpoint_interpolation_refiner_keeps_root_and_empty_fallback():
    model = EndpointInterpolationRefiner()
    base = torch.randn(2, 10, 23, 3)
    poses = torch.randn(2, 3, 23, 3)
    history = torch.randn(2, 6, 23, 3)
    valid = torch.tensor([[True, True, False], [False, False, False]])
    prediction, _ = model(base, poses, valid, history)
    assert torch.equal(prediction[:, :, 0], base[:, :, 0])
    assert torch.equal(prediction[1], base[1])
    prediction.square().mean().backward()
    assert model.mix_logit.grad is not None


def test_endpoint_terminal_refiner_uses_gaze_context_and_preserves_root():
    for model in (EndpointTerminalRefiner(), EndpointLocalMotionRefiner()):
        base = torch.randn(2, 10, 23, 3)
        poses = torch.randn(2, 3, 23, 3)
        history = torch.randn(2, 6, 23, 3)
        gaze = torch.randn(2, 6, 1, 3)
        context = torch.randn(2, 256)
        valid = torch.tensor([[True, True, False], [False, False, False]])
        prediction, _ = model(base, poses, valid, history, gaze, context)
        assert torch.equal(prediction[:, :, 0], base[:, :, 0])
        assert torch.equal(prediction[1], base[1])
        prediction.square().mean().backward()
        last = model.terminal[-1] if isinstance(model, EndpointTerminalRefiner) else model.output
        assert last.weight.grad is not None
