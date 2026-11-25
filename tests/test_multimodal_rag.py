import torch
import pytest

from model.multimodal_rag import (
    MotionBank,
    MotionBankItem,
    MotionRefiner,
    MultiModalRetriever,
    PrototypeConditionAdapter,
    PrototypeEncoder,
    audit_bank_items,
    from_reference,
    reference_frame,
    to_reference,
)


def _item(i, sequence=None, split="train"):
    g = torch.randn(6, 3)
    h = torch.randn(6, 23, 3)
    f = torch.randn(10, 23, 3)
    return MotionBankItem(
        sample_id=f"s{i}", dataset_name="GIMO", split=split,
        sequence_id=sequence or f"seq{i}", scene_id=f"scene{i}",
        history_frames=tuple(range(i, i + 6)), future_frames=tuple(range(i + 6, i + 16)),
        history_motion=torch.randn(6, 38), history_joints=h, history_gaze=g,
        future_motion=torch.randn(10, 38), future_joints=f,
        future_valid_mask=torch.ones(10, dtype=torch.bool), scene=torch.randn(20, 3),
    )


def test_coordinate_round_trip_and_bank_retrieval_padding():
    h = torch.randn(6, 23, 3)
    x = torch.randn(10, 23, 3)
    origin, basis = reference_frame(h)
    assert torch.allclose(from_reference(to_reference(x, origin, basis), origin, basis), x, atol=1e-5)
    bank = MotionBank([_item(0), _item(1)])
    bank.keys = torch.eye(2, 4)
    out = bank.retrieve(torch.tensor([[1., 0., 0., 0.]]), top_k=5)
    assert out["retrieved_ids"].shape == (1, 5)
    assert out["candidate_valid_mask"].tolist() == [[True, True, False, False, False]]
    assert out["retrieved_ids"][0, 2:].eq(-1).all()
    bank_same = MotionBank([_item(0, "same"), _item(1, "same")])
    bank_same.keys = torch.eye(2, 4)
    empty = bank_same.retrieve(torch.tensor([[1., 0., 0., 0.]]), top_k=3, metadata=[{"sequence_id": "same"}])
    assert not empty["candidate_valid_mask"][0].any() and empty["retrieved_ids"][0].eq(-1).all()


def test_bank_rejects_non_train_and_cross_split_overlap():
    with pytest.raises(ValueError):
        MotionBank([_item(0, split="test")])
    a, b = _item(0, "same", "train"), _item(1, "same", "test")
    with pytest.raises(ValueError):
        audit_bank_items([a, b], require_train=False)


def test_retriever_and_modules_are_finite_and_have_gradients():
    B, K, T, J = 2, 3, 10, 23
    motion = torch.randn(B, 6, 38, requires_grad=True)
    joints = torch.randn(B, 6, J, 3)
    gaze = torch.zeros(B, 6, 3)  # unknown/zero gaze is a supported finite path
    scene = torch.randn(B, 30, 3)
    retriever = MultiModalRetriever()
    query = retriever(motion, joints, gaze, scene)
    assert torch.isfinite(query).all()
    query.sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in retriever.parameters())

    proto = PrototypeEncoder()(torch.randn(B, K, T, J, 3))
    adapter = PrototypeConditionAdapter()
    guided, gate = adapter(torch.randn(B, 2, 256), proto, torch.zeros(B, K, dtype=torch.bool))
    assert torch.isfinite(guided).all() and torch.isfinite(gate).all()
    guided.sum().backward()
    assert adapter.out_proj.weight.grad is not None
    refiner = MotionRefiner()
    base = torch.randn(B, T, J, 3)
    refined, _, delta = refiner(motion.detach(), base, proto.detach(), query.detach(), torch.full((B, K), -float("inf")), torch.zeros(B, K, dtype=torch.bool))
    assert torch.isfinite(refined).all() and delta[:, :, 0].abs().max() == 0
    assert torch.allclose(refined, base)
    refined.sum().backward()
    assert refiner.delta.weight.grad is not None
    with pytest.raises(ValueError, match="history_motion"):
        refiner(torch.randn(B, 6, 256), base, proto.detach(), query.detach(), None, None)


def test_query_does_not_read_future_label():
    torch.manual_seed(4)
    model = MultiModalRetriever()
    model.eval()
    inputs = (torch.randn(1, 6, 38), torch.randn(1, 6, 23, 3), torch.randn(1, 6, 3), torch.randn(1, 16, 3))
    q1 = model(*inputs)
    future_a = torch.randn(1, 10, 23, 3)
    future_b = future_a + 1000
    # Future labels are intentionally absent from the encoder call.
    q2 = model(*inputs)
    assert torch.equal(q1, q2)
    assert not torch.equal(future_a, future_b)


def test_trained_adapter_has_exact_empty_candidate_fallback():
    adapter = PrototypeConditionAdapter(hidden_dim=32, prototype_dim=32).eval()
    torch.nn.init.normal_(adapter.out_proj.weight)
    torch.nn.init.normal_(adapter.out_proj.bias)
    hidden = torch.randn(2,10,32)
    output,_=adapter(hidden,torch.randn(2,4,10,32),torch.zeros(2,4,dtype=torch.bool))
    assert torch.equal(output,hidden)


def test_local_refiner_translation_rotation_equivariance_and_pelvis():
    from model.multimodal_rag import LocalMotionRefiner
    from utils.rotation import axis_angle_to_matrix,matrix_to_axis_angle
    model=LocalMotionRefiner(context_dim=32,prototype_dim=32,hidden_dim=32).eval()
    torch.nn.init.normal_(model.core.delta.weight)
    history=torch.randn(2,6,23,3);motion=torch.randn(2,6,38)
    base=torch.randn(2,10,23,3);proto=torch.randn(2,3,10,32);context=torch.randn(2,32)
    scores=torch.tensor([[.8,.3,float('-inf')],[.5,.2,float('-inf')]])
    valid=torch.tensor([[True,True,False],[True,True,False]])
    prediction,_,_=model(motion,base,proto,context,scores,valid,history_joints=history)
    assert torch.equal(prediction[:,:,0],base[:,:,0])
    rotation=axis_angle_to_matrix(torch.tensor([0.,1.1,0.]));shift=torch.tensor([100.,2.,-50.])
    shifted_motion=motion.clone()
    shifted_motion[...,:3]=matrix_to_axis_angle(rotation@axis_angle_to_matrix(motion[...,:3]))
    shifted_motion[...,3:6]=motion[...,3:6]@rotation.T+shift
    shifted,_,_=model(shifted_motion,base@rotation.T+shift,proto,context,scores,valid,
                      history_joints=history@rotation.T+shift)
    assert torch.allclose(shifted,prediction@rotation.T+shift,atol=2e-5)
    proto[:,2]+=10000
    masked,_,_=model(motion,base,proto,context,scores,valid,history_joints=history)
    assert torch.equal(masked,prediction)


def test_temporal_retriever_distinguishes_order_and_legacy_loads():
    model=MultiModalRetriever(dropout=0.0, temporal=True).eval()
    motion=torch.randn(2,6,38);history=torch.randn(2,6,23,3);gaze=torch.randn(2,6,3);scene=torch.randn(2,20,3)
    initial=model(motion,history,gaze,scene)
    reverse=model(motion.flip(1),history.flip(1),gaze.flip(1),scene)
    assert not torch.allclose(initial,reverse)
    assert 'time_embedding' in model.state_dict()
    legacy=MultiModalRetriever(temporal=False,dropout=0.0)
    assert 'time_embedding' not in legacy.state_dict()


def test_bank_serialization_saves_window_not_backing_cache(tmp_path):
    backing=torch.randn(100,1000,3)
    item=_item(0)
    item.scene=backing[0]
    bank=MotionBank([item])
    output=tmp_path/'bank.pt';bank.save(output)
    assert output.stat().st_size < 100000
    loaded=MotionBank.load(output)
    assert torch.equal(loaded.items[0].scene,backing[0])


def test_score_weighting_and_invalid_prototypes_do_not_change_legacy_refiner():
    model=MotionRefiner(context_dim=32,prototype_dim=32,hidden_dim=32).eval()
    torch.nn.init.normal_(model.delta.weight)
    hist=torch.randn(2,6,38);base=torch.randn(2,10,23,3)
    proto=torch.randn(2,3,10,32);valid=torch.tensor([[True,False,False],[True,False,False]])
    score=torch.tensor([[.3,-float('inf'),-float('inf')],[.4,-float('inf'),-float('inf')]])
    first=model(hist,base,proto,scores=score,candidate_valid_mask=valid)[0]
    proto[:,1:]+=10000
    second=model(hist,base,proto,scores=score,candidate_valid_mask=valid)[0]
    assert torch.equal(first,second)
    assert torch.equal(first[:,:,0],base[:,:,0])


def test_forecast_reranking_uses_predicted_pose_and_masks_invalid_candidates():
    from model.multimodal_rag import forecast_candidate_scores
    base=torch.randn(2,10,23,3)
    candidates=base[:,None].expand(-1,3,-1,-1,-1).clone()
    candidates[:,1,:,1:]+=1
    candidates[:,2,:,1:]+=100
    valid=torch.tensor([[True,True,False],[True,True,False]])
    scores=forecast_candidate_scores(base,candidates,valid)
    assert scores[:,0].eq(0).all() and scores[:,1].lt(0).all()
    assert scores[:,2].eq(-torch.inf).all()
    shift=torch.tensor([100.,-10.,4.])
    shifted=forecast_candidate_scores(base+shift,candidates+shift,valid)
    assert torch.allclose(scores,shifted,atol=1e-4)


def test_bank_reload_preserves_key_hash_bits(tmp_path):
    bank=MotionBank([_item(0),_item(1)])
    bank.encode_keys(MultiModalRetriever(dropout=0.0))
    original=bank.keys.clone();original_hash=bank.encoder_hash()
    bank.save(tmp_path/'bank.pt')
    restored=MotionBank.load(tmp_path/'bank.pt')
    assert torch.equal(original,restored.keys)
    assert original_hash==restored.encoder_hash()==restored.manifest['key_hash']


def test_prototype_blend_preserves_pelvis_and_has_learnable_trust():
    from model.multimodal_rag import PrototypeBlendRefiner
    refiner=PrototypeBlendRefiner()
    base=torch.randn(2,10,23,3);future=torch.randn(2,3,10,23,3);hist=torch.randn(2,6,38)
    valid=torch.tensor([[True,True,False],[False,False,False]])
    initial,_,_=refiner(hist,base,candidate_valid_mask=valid,prototype_future=future)
    assert torch.equal(initial,base)
    initial.square().mean().backward()
    assert refiner.mix.grad is not None and refiner.mix.grad.abs()>0
    refiner.mix.data.fill_(.3)
    prediction,_,_=refiner(hist,base,candidate_valid_mask=valid,prototype_future=future)
    assert torch.equal(prediction[:,:,0],base[:,:,0])
    assert torch.equal(prediction[1],base[1])
    expected=.7*(base[0]-base[0,:,:1])+.3*(future[0,:2]-future[0,:2,:,:1]).mean(0)+base[0,:,:1]
    assert torch.allclose(prediction[0],expected,atol=1e-6)
    future[:,2]+=10000
    repeated,_,_=refiner(hist,base,candidate_valid_mask=valid,prototype_future=future)
    assert torch.equal(prediction,repeated)


def test_endpoint_guidance_preserves_trajectory_and_early_frames():
    from model.multimodal_rag import EndpointPrototypeRefiner
    model=EndpointPrototypeRefiner(endpoint_topk=2,endpoint_frames=4)
    base=torch.randn(2,10,23,3);proto=torch.randn(2,3,10,23,3);hist=torch.randn(2,6,38)
    valid=torch.tensor([[True,True,False],[True,True,False]])
    endpoints=torch.randn(2,5,23,3);endpoint_valid=torch.tensor([[True,True,False,False,False],[False,False,False,False,False]])
    baseline,_,_=model(hist,base,candidate_valid_mask=valid,prototype_future=proto,
                       endpoint_future=endpoints,endpoint_valid=endpoint_valid)
    baseline.square().mean().backward()
    assert model.endpoint_mix.grad is not None and model.endpoint_mix.grad.abs().sum()>0
    assert model.mix.grad is None
    model.endpoint_mix.data.fill_(.3)
    prediction,_,correction=model(hist,base,candidate_valid_mask=valid,prototype_future=proto,
                                  endpoint_future=endpoints,endpoint_valid=endpoint_valid)
    assert torch.equal(prediction[:,:6],baseline[:,:6])
    assert torch.equal(prediction[:,:,0],base[:,:,0])
    assert torch.equal(prediction[1],baseline[1])
    assert torch.allclose(correction[:, -4]*4,correction[:, -1],atol=1e-6)
    endpoints[:,2:]+=10000
    repeated,_,_=model(hist,base,candidate_valid_mask=valid,prototype_future=proto,
                        endpoint_future=endpoints,endpoint_valid=endpoint_valid)
    assert torch.equal(repeated,prediction)


def test_endpoint_bank_excludes_query_recording_and_reads_training_futures_only():
    from model.multimodal_rag import endpoint_candidates
    bank=MotionBank([_item(0),_item(1)])
    histories=torch.randn(2,6,23,3)
    endpoints,valid=endpoint_candidates(bank,histories,[{'sequence_id':'seq0'},{'sequence_id':'seq1'}])
    assert endpoints.shape==(2,2,23,3)
    assert valid.tolist()==[[False,True],[True,False]]
    with pytest.raises(ValueError,match='metadata'):
        endpoint_candidates(bank,histories,None)


def test_precomputed_observation_features_preserve_encoder_output_and_gradients():
    torch.manual_seed(42)
    model=MultiModalRetriever(dropout=0.0).eval()
    inputs=(torch.randn(2,6,38),torch.randn(2,6,23,3),torch.randn(2,6,1,3),torch.randn(2,30,3))
    direct=model(*inputs)
    features=model.prepare_features(*inputs)
    cached=model.encode_features(*features)
    assert torch.equal(direct,cached)
    direct.square().sum().backward()
    gradients={name:p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.encode_features(*features).square().sum().backward()
    assert all(torch.equal(gradients[name],p.grad) for name,p in model.named_parameters() if name in gradients)


def test_canonical_future_distance_matches_history_aligned_target_distance():
    from model.multimodal_rag import align_future_to_query
    a,b=_item(0),_item(1)
    aligned=align_future_to_query(a.future_joints,a.history_joints,b.history_joints)
    direct=((aligned-aligned[:,:1])-(b.future_joints-b.future_joints[:,:1])).norm(dim=-1).mean()
    aa=(a.future_joints-a.future_joints[:,:1])@reference_frame(a.history_joints)[1]
    bb=(b.future_joints-b.future_joints[:,:1])@reference_frame(b.history_joints)[1]
    cached=(aa-bb).norm(dim=-1).mean()
    assert torch.allclose(direct,cached,atol=1e-6)


def test_recording_distinct_candidates_skip_duplicate_source_windows():
    bank=MotionBank([_item(0,'a'),_item(1,'a'),_item(2,'b'),_item(3,'c')])
    bank.keys=torch.nn.functional.normalize(torch.tensor([[1.,0.],[.99,.01],[.8,.2],[.7,.3]]),dim=-1)
    query=torch.tensor([[1.,0.]])
    result=bank.retrieve(query,5,unique_sequences=True)
    ids=result['retrieved_ids'][0].tolist()
    assert ids[:3]==[0,2,3] and ids[3:]==[-1,-1]
    assert result['candidate_valid_mask'][0].tolist()==[True,True,True,False,False]


def test_full_bank_terminal_supervision_has_finite_masked_gradients():
    from scripts.train_terminal_retriever import full_bank_terminal_loss
    similarity=torch.tensor([[.7,.2,.9],[.1,.8,.3]],requires_grad=True)
    distance=torch.tensor([[.05,.2,0.],[.4,0.,.1]])
    legal=torch.tensor([[True,True,False],[True,False,True]])
    loss=full_bank_terminal_loss(similarity,distance,legal,.05)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(similarity.grad).all()
    assert similarity.grad[~legal].eq(0).all()
    altered=distance.clone();altered[~legal]=10000
    assert torch.equal(loss.detach(),full_bank_terminal_loss(similarity.detach(),altered,legal,.05))
    with pytest.raises(ValueError,match='legal'):
        full_bank_terminal_loss(similarity,distance,torch.zeros_like(legal))


def test_rich_observed_summary_starts_from_pooled_encoder_and_can_learn_changes():
    from scripts.upgrade_retrieval_summary import upgrade_encoder
    torch.manual_seed(42)
    encoder=MultiModalRetriever(dropout=0.0).eval()
    upgraded=upgrade_encoder(encoder).eval()
    inputs=(torch.randn(2,6,38),torch.randn(2,6,23,3),torch.randn(2,6,1,3),torch.randn(2,32,3))
    assert torch.allclose(encoder(*inputs),upgraded(*inputs),atol=1e-7)
    loss=(upgraded(*inputs)*torch.randn(2,256)).sum();loss.backward()
    old_dim=encoder.motion_branch[0].in_features
    assert upgraded.motion_branch[0].weight.grad[:,old_dim:].abs().sum()>0
    assert upgraded.encoder_version=='multimodal-retriever-v3'
    with pytest.raises(ValueError,match='either'):
        MultiModalRetriever(temporal=True,rich_summary=True)


def test_rich_summary_bank_round_trip(tmp_path):
    from model.multimodal_rag import retriever_from_bank
    bank=MotionBank([_item(0),_item(1)])
    encoder=MultiModalRetriever(dropout=0.0,rich_summary=True).eval()
    bank.encode_keys(encoder);bank.manifest.update(use_motion=True,use_gaze=True,use_scene=True)
    bank.save(tmp_path/'bank.pt');restored=MotionBank.load(tmp_path/'bank.pt')
    queried=retriever_from_bank(restored).eval()
    item=bank.items[0];inputs=(item.history_motion[None],item.history_joints[None],item.history_gaze[None],item.scene[None])
    assert torch.equal(encoder(*inputs),queried(*inputs))


def test_same_recording_candidates_require_disjoint_frame_intervals():
    import dataclasses
    overlap=dataclasses.replace(_item(0,'query'),sample_id='query:100',history_frames=tuple(range(100,106)),future_frames=tuple(range(106,116)))
    disjoint=dataclasses.replace(_item(1,'query'),sample_id='query:0',history_frames=tuple(range(6)),future_frames=tuple(range(6,16)))
    other=_item(2,'other')
    bank=MotionBank([overlap,disjoint,other],keys=torch.tensor([[1.,0.],[1.,0.],[0.,1.]]))
    result=bank.retrieve(torch.tensor([[1.,0.]]),3,metadata=[{'sequence_id':'query','history_frames':[90,95],'future_frames':[96,105]}],exclude_same_sequence=False)
    assert result['retrieved_ids'][0].tolist()==[1,2,-1]
    with pytest.raises(ValueError,match='frame interval'):
        bank.retrieve(torch.tensor([[1.,0.]]),3,metadata=[{'sequence_id':'query'}],exclude_same_sequence=False)


def test_same_scene_priority_preserves_legal_filters():
    import dataclasses
    foreign=dataclasses.replace(_item(0,'foreign'),scene_id='other')
    same=dataclasses.replace(_item(1,'same'),scene_id='room')
    query=dataclasses.replace(_item(2,'query'),scene_id='room')
    bank=MotionBank([foreign,same,query],keys=torch.tensor([[1.,0.],[0.,1.],[1.,0.]]))
    result=bank.retrieve(torch.tensor([[1.,0.]]),3,metadata=[{'sequence_id':'query','scene_id':'room'}],prefer_same_scene=True)
    assert result['retrieved_ids'][0].tolist()==[1,0,-1]
    assert not result['candidate_valid_mask'][0,2]


def test_vectorized_endpoint_alignment_matches_window_alignment():
    from model.multimodal_rag import endpoint_candidates,align_future_to_query
    bank=MotionBank([_item(0),_item(1),_item(2)])
    history=torch.randn(2,6,23,3)
    endpoints,_=endpoint_candidates(bank,history,[{'sequence_id':'q0'},{'sequence_id':'q1'}])
    direct=torch.stack([torch.stack([align_future_to_query(item.future_joints[-1:],item.history_joints,row)[0] for item in bank.items]) for row in history])
    assert torch.allclose(endpoints,direct,atol=1e-6)
    assert torch.equal(endpoints,endpoint_candidates(bank,history,[{'sequence_id':'q0'},{'sequence_id':'q1'}])[0])


def test_joint_terminal_blend_starts_from_base_and_preserves_root_and_prefix():
    from model.multimodal_rag import TerminalBlendRefiner,PrototypeBlendRefiner
    model=TerminalBlendRefiner(endpoint_frames=4)
    baseline=PrototypeBlendRefiner();baseline.mix.data.copy_(model.mix.data)
    hist=torch.randn(2,6,38);base=torch.randn(2,10,23,3);future=torch.randn(2,3,10,23,3)
    valid=torch.tensor([[True,True,False],[False,False,False]])
    before,_,_=baseline(hist,base,candidate_valid_mask=valid,prototype_future=future)
    initial,_,_=model(hist,base,candidate_valid_mask=valid,prototype_future=future)
    assert torch.equal(before,initial)
    initial.square().mean().backward()
    assert model.terminal_mix.grad.abs().sum()>0 and model.mix.grad is None
    model.terminal_mix.data.fill_(.5)
    changed,_,delta=model(hist,base,candidate_valid_mask=valid,prototype_future=future)
    assert torch.equal(changed[:,:6],before[:,:6])
    assert torch.equal(changed[:,:,0],base[:,:,0])
    assert torch.equal(changed[1],base[1])
    assert delta[:, :6].eq(0).all()
    future[:,2]+=10000
    again,_,_=model(hist,base,candidate_valid_mask=valid,prototype_future=future)
    assert torch.equal(changed,again)


def test_adaptive_terminal_trust_is_observation_only_and_root_preserving():
    from model.multimodal_rag import AdaptiveTerminalRefiner,PrototypeBlendRefiner
    model=AdaptiveTerminalRefiner(hidden_dim=0)
    base=torch.randn(2,10,23,3);history=torch.randn(2,6,23,3);motion=torch.randn(2,6,38)
    future=torch.randn(2,3,10,23,3);valid=torch.tensor([[True,True,False],[False,False,False]])
    baseline=PrototypeBlendRefiner();baseline.mix.data.copy_(model.mix.data)
    expected=baseline(motion,base,candidate_valid_mask=valid,prototype_future=future)[0]
    initial=model(motion,base,candidate_valid_mask=valid,prototype_future=future,history_joints=history)[0]
    assert torch.allclose(initial,expected,atol=1e-6)
    initial.square().mean().backward()
    assert model.trust[0].weight.grad.abs().sum()>0
    assert model.mix.grad is None
    torch.nn.init.normal_(model.trust[0].weight)
    updated=model(motion,base,candidate_valid_mask=valid,prototype_future=future,history_joints=history)[0]
    assert torch.equal(updated[:,:,0],base[:,:,0])
    assert torch.allclose(updated[:,:6],expected[:,:6],atol=1e-6)
    assert torch.equal(updated[1],base[1])
    future[:,2]+=10000
    same=model(motion,base,candidate_valid_mask=valid,prototype_future=future,history_joints=history)[0]
    assert torch.equal(same,updated)


def test_terminal_fusion_supervision_preserves_masked_candidate_gradients():
    from scripts.train_terminal_retriever import terminal_fusion_loss
    similarity=torch.randn(3,3,requires_grad=True)
    endpoints=torch.randn(3,23,3)
    legal=~torch.eye(3,dtype=torch.bool)
    loss=terminal_fusion_loss(similarity,endpoints,legal)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(similarity.grad).all()
    assert similarity.grad[~legal].eq(0).all()
    assert similarity.grad[legal].abs().sum()>0
    with pytest.raises(ValueError,match='legal'):
        terminal_fusion_loss(similarity,endpoints,torch.zeros_like(legal))


def test_body_part_trust_shares_weights_and_preserves_pelvis():
    from model.multimodal_rag import AdaptiveTerminalRefiner
    model=AdaptiveTerminalRefiner(hidden_dim=0,part_trust=True)
    assert model.trust[-1].out_features==5
    assert model.part_index.tolist()==[0,1,2,0,1,2,0,1,2,0,1,2,0,3,4,0,3,4,3,4,3,4,0]
    model.trust[-1].bias.data.copy_(torch.arange(5).float())
    base=torch.randn(2,10,23,3);future=torch.randn(2,3,10,23,3);history=torch.randn(2,6,23,3);valid=torch.ones(2,3,dtype=torch.bool)
    result,_,_=model(torch.randn(2,6,38),base,candidate_valid_mask=valid,prototype_future=future,history_joints=history)
    assert torch.equal(result[:,:,0],base[:,:,0])
    result.square().mean().backward()
    assert model.trust[-1].weight.grad.abs().sum()>0
    with pytest.raises(ValueError,match='not both'):
        AdaptiveTerminalRefiner(joint_trust=True,part_trust=True)


def test_extended_trust_limit_is_serialized_and_initializes_at_baseline_mix():
    from model.multimodal_rag import AdaptiveTerminalRefiner,PrototypeBlendRefiner
    model=AdaptiveTerminalRefiner(hidden_dim=0,max_trust=1.2)
    assert 'trust_limit' in model.state_dict()
    assert torch.allclose(model.trust_limit,torch.tensor(1.2))
    motion=torch.randn(2,6,38);base=torch.randn(2,10,23,3);future=torch.randn(2,3,10,23,3);hist=torch.randn(2,6,23,3);valid=torch.ones(2,3,dtype=torch.bool)
    baseline=PrototypeBlendRefiner();baseline.mix.data.copy_(model.mix.data)
    expected=baseline(motion,base,candidate_valid_mask=valid,prototype_future=future)[0]
    initial=model(motion,base,candidate_valid_mask=valid,prototype_future=future,history_joints=hist)[0]
    assert torch.allclose(initial,expected,atol=1e-6)
    model.trust[-1].bias.data.fill_(10.)
    changed,trust,_=model(motion,base,candidate_valid_mask=valid,prototype_future=future,history_joints=hist)
    assert trust.gt(1.).all() and trust.le(1.2).all()
    assert torch.equal(changed[:,:,0],base[:,:,0])
    assert torch.allclose(changed[:,:6],expected[:,:6],atol=1e-6)
    restored=AdaptiveTerminalRefiner(hidden_dim=0,max_trust=1.2);restored.load_state_dict(model.state_dict(),strict=True)
    assert torch.equal(changed,restored(motion,base,candidate_valid_mask=valid,prototype_future=future,history_joints=hist)[0])
    assert 'trust_limit' not in AdaptiveTerminalRefiner().state_dict()
    with pytest.raises(ValueError,match='trust limit'):
        AdaptiveTerminalRefiner(max_trust=2.)
