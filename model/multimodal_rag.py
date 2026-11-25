"""Small, local retrieval stack for GIMO motion forecasting.

The module deliberately keeps retrieval independent from the SIF3D backbone.  A
bank stores complete training windows, while the key is produced only from the
observed motion, gaze, and scene.  No MoRAG code is imported here: MoRAG is a
reference for normalized retrieval only; the encoders and prototype path are
new implementation for this repository.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from utils.rotation import axis_angle_to_matrix, matrix_to_axis_angle


BANK_VERSION = "gimo-motion-bank-v1"
ENCODER_VERSION = "multimodal-retriever-v2"


def _finite(x: Tensor) -> Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _as_gaze(gaze: Tensor) -> Tensor:
    if gaze.ndim == 4:
        if gaze.shape[2] != 1:
            raise ValueError(f"gaze must have one point per frame, got {tuple(gaze.shape)}")
        gaze = gaze[:, :, 0]
    if gaze.ndim != 3 or gaze.shape[-1] != 3:
        raise ValueError(f"gaze must be [B,T,3] or [B,T,1,3], got {tuple(gaze.shape)}")
    return _finite(gaze)


def _normalise_mask(mask: Optional[Tensor], shape: Tuple[int, int], device: torch.device) -> Tensor:
    if mask is None:
        # Unknown validity is represented as zero in the feature, never as
        # fabricated "all valid" data.
        return torch.zeros(shape, dtype=torch.bool, device=device)
    mask = mask.to(device=device, dtype=torch.bool)
    if tuple(mask.shape) != shape:
        raise ValueError(f"gaze_valid must be {shape}, got {tuple(mask.shape)}")
    return mask


def reference_frame(history_joints: Tensor, pelvis_index: int = 0) -> Tuple[Tensor, Tensor]:
    """Return origin and a fixed world-to-reference basis from history only.

    GIMO uses y as vertical.  The horizontal heading is estimated from the
    last-frame shoulder/hip width and falls back to observed pelvis velocity.
    The basis columns are local right, up, and forward axes in world space.
    """
    if history_joints.ndim != 3 or history_joints.shape[-1] != 3:
        raise ValueError("history_joints must be [T,J,3]")
    h = _finite(history_joints)
    origin = h[-1, pelvis_index]
    up = h.new_tensor([0.0, 1.0, 0.0])
    right = None
    for a, b in ((2, 1), (16, 15), (1, 2)):
        if max(a, b) < h.shape[1]:
            v = h[-1, a] - h[-1, b]
            if torch.linalg.vector_norm(v[[0, 2]]) > 1e-5:
                right = F.normalize(v * torch.tensor([1.0, 0.0, 1.0], device=h.device, dtype=h.dtype), dim=0)
                break
    if right is None:
        velocity = h[-1, pelvis_index] - h[0, pelvis_index]
        if torch.linalg.vector_norm(velocity[[0, 2]]) > 1e-5:
            forward = F.normalize(velocity * torch.tensor([1.0, 0.0, 1.0], device=h.device, dtype=h.dtype), dim=0)
            right = F.normalize(torch.linalg.cross(up, forward, dim=0), dim=0)
        else:
            right = h.new_tensor([1.0, 0.0, 0.0])
    forward = F.normalize(torch.linalg.cross(right, up, dim=0), dim=0)
    right = F.normalize(torch.linalg.cross(up, forward, dim=0), dim=0)
    basis = torch.stack((right, up, forward), dim=-1)
    return origin, basis


def to_reference(points: Tensor, origin: Tensor, basis: Tensor) -> Tensor:
    return (_finite(points) - origin) @ basis


def from_reference(points: Tensor, origin: Tensor, basis: Tensor) -> Tensor:
    return _finite(points) @ basis.transpose(-1, -2) + origin


def align_future_to_query(
    prototype_future: Tensor,
    prototype_history: Tensor,
    query_history: Tensor,
    pelvis_index: int = 0,
) -> Tensor:
    """Align one prototype using the two histories, never the query future."""
    p_origin, p_basis = reference_frame(prototype_history, pelvis_index)
    q_origin, q_basis = reference_frame(query_history, pelvis_index)
    return from_reference(to_reference(prototype_future, p_origin, p_basis), q_origin, q_basis)


@dataclasses.dataclass
class MotionBankItem:
    sample_id: str
    dataset_name: str
    split: str
    sequence_id: str
    scene_id: str
    history_frames: Tuple[int, ...]
    future_frames: Tuple[int, ...]
    history_motion: Tensor
    history_joints: Tensor
    history_gaze: Tensor
    future_motion: Tensor
    future_joints: Tensor
    future_valid_mask: Tensor
    scene: Tensor
    gaze_valid: Optional[Tensor] = None
    transform_origin: Optional[Tensor] = None
    transform_basis: Optional[Tensor] = None
    retrieval_key: Optional[Tensor] = None
    encoder_version: str = ENCODER_VERSION

    def __post_init__(self) -> None:
        self.split = str(self.split).lower()
        if self.history_joints.ndim != 3 or self.future_joints.ndim != 3:
            raise ValueError("history_joints and future_joints must be [T,J,3]")
        if self.history_motion.shape[0] != self.history_joints.shape[0]:
            raise ValueError("history motion/joints lengths differ")
        if self.future_motion.shape[0] != self.future_joints.shape[0]:
            raise ValueError("future motion/joints lengths differ")
        if self.future_valid_mask.shape != (self.future_joints.shape[0],):
            raise ValueError("future_valid_mask must be [T_future]")
        if self.scene.ndim != 2 or self.scene.shape[-1] != 3:
            raise ValueError("scene must be [N,3]")
        if self.transform_origin is None or self.transform_basis is None:
            origin, basis = reference_frame(self.history_joints)
            self.transform_origin, self.transform_basis = origin.cpu(), basis.cpu()

    def metadata(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "sequence_id": self.sequence_id,
            "scene_id": self.scene_id,
            "history_frames": list(self.history_frames),
            "future_frames": list(self.future_frames),
            "encoder_version": self.encoder_version,
        }

    def to_dict(self) -> Dict[str, Any]:
        # asdict deep-copies tensor views and serializes their entire backing cache per item.
        out = {field.name: (getattr(self, field.name).detach().cpu().clone()
                            if isinstance(getattr(self, field.name), Tensor) else getattr(self, field.name))
               for field in dataclasses.fields(self)}
        out["metadata"] = self.metadata()
        return out


def _interval_overlap(a: Sequence[int], b: Sequence[int]) -> bool:
    if not a or not b:
        return False
    return max(a[0], b[0]) <= min(a[-1], b[-1])


def audit_bank_items(items: Sequence[MotionBankItem], require_train: bool = True) -> None:
    if not items:
        raise ValueError("motion bank is empty")
    if require_train and any(i.split != "train" for i in items):
        bad = sorted({i.split for i in items if i.split != "train"})
        raise ValueError(f"bank requires an explicit train manifest; found split(s): {bad}")
    seen_ids = set()
    for item in items:
        if item.sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id in bank: {item.sample_id}")
        seen_ids.add(item.sample_id)
    by_sequence: Dict[str, List[MotionBankItem]] = {}
    for item in items:
        by_sequence.setdefault(item.sequence_id, []).append(item)
    for sequence_items in by_sequence.values():
        for left in sequence_items:
            for right in sequence_items:
                if left is right or left.split == right.split:
                    continue
                if _interval_overlap(left.history_frames + left.future_frames, right.history_frames + right.future_frames):
                    raise ValueError(f"cross-split frame overlap: {left.sample_id} and {right.sample_id}")


class MotionBank:
    def __init__(self, items: Sequence[MotionBankItem], keys: Optional[Tensor] = None, manifest: Optional[Mapping[str, Any]] = None,
                 encoder_state_dict: Optional[Mapping[str, Tensor]] = None):
        audit_bank_items(items, require_train=True)
        self.items = list(items)
        self.manifest = dict(manifest or {})
        self.manifest.setdefault("bank_version", BANK_VERSION)
        self.manifest.setdefault("encoder_version", ENCODER_VERSION)
        if keys is not None and (keys.ndim != 2 or not torch.isfinite(keys).all() or
                                 not torch.allclose(keys.norm(dim=-1), torch.ones(keys.shape[0], device=keys.device), atol=1e-5)):
            raise ValueError('Bank keys must be finite normalized [N,D]')
        # A second normalization changes float bits and invalidates the saved key hash.
        self.keys = None if keys is None else keys.float().cpu()
        self.encoder_state_dict = None if encoder_state_dict is None else {k: v.cpu() for k, v in encoder_state_dict.items()}
        if self.keys is not None and self.keys.shape[0] != len(self.items):
            raise ValueError("key count does not match bank item count")

    @classmethod
    def from_gimo_dataset(cls, dataset: Any) -> "MotionBank":
        if not getattr(dataset, "train", False):
            raise ValueError("MotionBank.from_gimo_dataset requires the explicit train dataset")
        items: List[MotionBankItem] = []
        for idx in range(len(dataset)):
            seq = str(dataset.sequences_path_list[idx])
            scene = str(dataset.scenes_path_list[idx])
            start, end = dataset.start_end_list[idx]
            hist = dataset.poses_input[idx].detach().cpu().float()
            future = dataset.poses_label[idx].detach().cpu().float()
            hist_j = dataset.joints_input[idx][:, :23].detach().cpu().float()
            future_j = dataset.joints_label[idx][:, :23].detach().cpu().float()
            # Cached GIMO files do not preserve the raw gaze-validity mask.  None
            # is intentional: the encoder records this as unknown.
            items.append(MotionBankItem(
                sample_id=f"{seq}:{int(start)}",
                dataset_name="GIMO",
                split="train",
                sequence_id=seq,
                scene_id=scene,
                history_frames=tuple(int(start) + int(f * 30 / dataset.fps) for f in range(dataset.input_seq_len)),
                future_frames=tuple(int(start) + int((dataset.input_seq_len + f) * 30 / dataset.fps) for f in range(dataset.output_seq_len)),
                history_motion=hist,
                history_joints=hist_j,
                history_gaze=dataset.gazes[idx].detach().cpu().float(),
                future_motion=future,
                future_joints=future_j,
                future_valid_mask=torch.ones(future_j.shape[0], dtype=torch.bool),
                scene=dataset.scenes[idx].detach().cpu().float(),
                gaze_valid=None,
            ))
        return cls(items)

    def __len__(self) -> int:
        return len(self.items)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": BANK_VERSION, "manifest": self.manifest, "items": [i.to_dict() for i in self.items], "keys": self.keys,
                   "encoder_state_dict": self.encoder_state_dict}
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "MotionBank":
        payload = torch.load(path, map_location=map_location)
        if payload.get('version') == 'endpoint-pose-bank-v1':
            from model.endpoint_pose_bank import EndpointPoseBank, EndpointPoseItem
            return EndpointPoseBank([EndpointPoseItem(**raw) for raw in payload['items']], payload.get('keys'),
                                    payload.get('manifest'), payload.get('encoder_state_dict'),
                                    payload.get('pose_encoder_state_dict'))
        if payload.get("version") != BANK_VERSION:
            raise ValueError(f"unsupported/corrupt motion bank version: {payload.get('version')!r}")
        items = []
        for raw in payload["items"]:
            raw.pop("metadata", None)
            items.append(MotionBankItem(**raw))
        return cls(items, payload.get("keys"), payload.get("manifest"), payload.get("encoder_state_dict"))

    def encoder_hash(self) -> str:
        if self.keys is None:
            return ""
        return hashlib.sha256(self.keys.detach().cpu().numpy().tobytes()).hexdigest()

    def encode_keys(self, encoder: "MultiModalRetriever", batch_size: int = 32, device: Optional[torch.device] = None) -> Tensor:
        device = device or next(encoder.parameters()).device
        chunks = []
        encoder.eval()
        with torch.no_grad():
            for start in range(0, len(self.items), batch_size):
                group = self.items[start:start + batch_size]
                gaze_valid = None
                if all(i.gaze_valid is not None for i in group):
                    gaze_valid = torch.stack([i.gaze_valid for i in group]).to(device)
                chunks.append(encoder(
                    torch.stack([i.history_motion for i in group]).to(device),
                    torch.stack([i.history_joints for i in group]).to(device),
                    torch.stack([i.history_gaze for i in group]).to(device),
                    torch.stack([i.scene for i in group]).to(device),
                    gaze_valid,
                ).cpu())
        self.keys = torch.cat(chunks, dim=0) if chunks else torch.empty(0, encoder.embedding_dim)
        self._keys_cache = {}
        self.encoder_state_dict = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
        for item, key in zip(self.items, self.keys):
            item.retrieval_key = key.clone()
            item.encoder_version = encoder.encoder_version
        self.manifest.update({"encoder_version": encoder.encoder_version, "embedding_dim": encoder.embedding_dim, "key_hash": self.encoder_hash()})
        return self.keys

    def legal_candidate_mask(self, metadata, exclude_same_sequence=True, exclude_scene=False, device=None):
        meta = metadata
        seq, scene, sample_id = (str(meta.get(key, '')) for key in ('sequence_id', 'scene_id', 'sample_id'))
        valid = torch.ones(len(self.items), dtype=torch.bool, device=device)
        if not exclude_same_sequence:
            frames = meta.get('history_frames', []) + meta.get('future_frames', [])
            if not frames and sample_id:
                try:
                    start_frame = int(sample_id.rsplit(':', 1)[1])
                except (IndexError, ValueError) as error:
                    raise ValueError('Same-recording retrieval requires a valid query frame interval') from error
                if 'input_seq_len' not in self.manifest or 'output_seq_len' not in self.manifest or 'fps' not in self.manifest:
                    raise ValueError('Frame filtering requires bank window lengths and FPS')
                span = int((self.manifest['input_seq_len'] + self.manifest['output_seq_len'] - 1) * 30 / self.manifest['fps'])
                frames = [start_frame, start_frame + span]
            if not seq or not frames:
                raise ValueError('Same-recording retrieval requires sequence_id and frame interval metadata')
            valid &= torch.tensor([item.sequence_id != seq or not _interval_overlap(
                frames, item.history_frames + item.future_frames) for item in self.items], device=device)
        if exclude_same_sequence and seq:
            valid &= torch.tensor([i.sequence_id != seq for i in self.items], device=device)
        if exclude_scene and scene:
            valid &= torch.tensor([i.scene_id != scene for i in self.items], device=device)
        if sample_id:
            valid &= torch.tensor([i.sample_id != sample_id for i in self.items], device=device)
        return valid

    def retrieve(self, query: Tensor, top_k: int = 5, metadata: Optional[Sequence[Mapping[str, Any]]] = None,
                 exclude_same_sequence: bool = True, exclude_scene: bool = False,
                 chunk_size: int = 4096, backend: str = "torch", unique_sequences: bool = False,
                 prefer_same_scene: bool = False) -> Dict[str, Tensor | List[List[Dict[str, Any]]]]:
        if self.keys is None:
            raise RuntimeError("motion bank has no keys; encode it with the frozen retriever first")
        if query.ndim != 2 or query.shape[1] != self.keys.shape[1]:
            raise ValueError(f"query must be [B,{self.keys.shape[1]}], got {tuple(query.shape)}")
        if metadata is not None and len(metadata) != query.shape[0]:
            raise ValueError("metadata length must match query batch size")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if backend not in ("torch", "faiss"):
            raise ValueError("backend must be 'torch' or 'faiss'")
        if backend == "faiss":
            if unique_sequences or prefer_same_scene or not exclude_same_sequence:
                raise ValueError('Diverse/scene-priority/frame-filtered candidates require the exact PyTorch backend')
            try:
                import faiss  # optional dependency
            except ImportError as exc:
                raise ImportError("FAISS backend requested but faiss is not installed") from exc
            # Apply the same metadata filters before searching so IDs remain
            # the bank's stable IDs. IndexFlatIP is exact for normalized keys.
            q_np = F.normalize(_finite(query.float()), dim=-1).detach().cpu().numpy()
            all_ids, all_scores, all_masks, audit = [], [], [], []
            for b in range(query.shape[0]):
                meta = metadata[b] if metadata is not None else {}
                seq, scene, sample_id = str(meta.get("sequence_id", "")), str(meta.get("scene_id", "")), str(meta.get("sample_id", ""))
                valid_cpu = torch.ones(len(self.items), dtype=torch.bool)
                if exclude_same_sequence and seq:
                    valid_cpu &= torch.tensor([i.sequence_id != seq for i in self.items])
                if exclude_scene and scene:
                    valid_cpu &= torch.tensor([i.scene_id != scene for i in self.items])
                if sample_id:
                    valid_cpu &= torch.tensor([i.sample_id != sample_id for i in self.items])
                allowed = torch.where(valid_cpu)[0].tolist()
                index = faiss.IndexFlatIP(self.keys.shape[1])
                if allowed:
                    index.add(self.keys[allowed].numpy())
                    take = min(top_k, len(allowed))
                    scores_np, local_np = index.search(q_np[b:b + 1], take)
                    ids = torch.full((top_k,), -1, dtype=torch.long)
                    scores = torch.full((top_k,), -torch.inf)
                    ids[:take] = torch.tensor([allowed[int(i)] for i in local_np[0]])
                    scores[:take] = torch.from_numpy(scores_np[0])
                    mask = torch.zeros(top_k, dtype=torch.bool); mask[:take] = True
                else:
                    ids = torch.full((top_k,), -1, dtype=torch.long); scores = torch.full((top_k,), -torch.inf); mask = torch.zeros(top_k, dtype=torch.bool)
                all_ids.append(ids.to(query.device)); all_scores.append(scores.to(query.device)); all_masks.append(mask.to(query.device))
                audit.append([{"candidate_count": len(allowed), "excluded_same_sequence": exclude_same_sequence, "excluded_scene": exclude_scene, "excluded_sample_id": bool(sample_id)}])
            return {"retrieved_ids": torch.stack(all_ids), "retrieval_scores": torch.stack(all_scores), "candidate_valid_mask": torch.stack(all_masks), "filter_audit": audit}
        q = F.normalize(_finite(query.float()), dim=-1)
        cache_key = (q.device.type, q.device.index, q.dtype)
        cache = getattr(self, '_keys_cache', {})
        if cache_key not in cache:
            cache[cache_key] = self.keys.to(device=q.device, dtype=q.dtype)
            self._keys_cache = cache
        keys = cache[cache_key]
        B, N, K = q.shape[0], keys.shape[0], min(top_k, keys.shape[0])
        all_ids, all_scores, all_masks, audit = [], [], [], []
        for b in range(B):
            valid = torch.ones(N, dtype=torch.bool, device=q.device)
            meta = metadata[b] if metadata is not None else {}
            seq = str(meta.get("sequence_id", ""))
            scene = str(meta.get("scene_id", ""))
            sample_id = str(meta.get("sample_id", ""))
            valid = self.legal_candidate_mask(meta, exclude_same_sequence, exclude_scene, q.device)
            values = torch.full((N,), -torch.inf, device=q.device, dtype=q.dtype)
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                values[start:end] = q[b:b + 1] @ keys[start:end].transpose(0, 1)
            values = values.masked_fill(~valid, -torch.inf)
            if prefer_same_scene:
                if not scene:
                    raise ValueError('Scene-priority retrieval requires query scene_id')
                values = values + torch.tensor([3.0 if item.scene_id == scene else 0.0 for item in self.items], device=q.device)
            if unique_sequences:
                selected, seen = [], set()
                for idx in values.argsort(descending=True).tolist():
                    if not torch.isfinite(values[idx]):
                        break
                    sequence = self.items[idx].sequence_id
                    if sequence in seen:
                        continue
                    selected.append(idx); seen.add(sequence)
                    if len(selected) == top_k:
                        break
                ids = torch.tensor(selected, device=q.device, dtype=torch.long)
                scores = values[ids]
                K = len(selected)
            elif K:
                scores, ids = torch.topk(values, K, dim=0)
            else:
                scores, ids = values.new_empty(0), torch.empty(0, dtype=torch.long, device=q.device)
            valid_top = torch.isfinite(scores)
            ids = ids.masked_fill(~valid_top, -1)
            pad = top_k - K
            if pad:
                ids = torch.cat([ids, torch.full((pad,), -1, dtype=torch.long, device=q.device)])
                scores = torch.cat([scores, torch.full((pad,), -torch.inf, dtype=q.dtype, device=q.device)])
                valid_top = torch.cat([valid_top, torch.zeros(pad, dtype=torch.bool, device=q.device)])
            all_ids.append(ids)
            all_scores.append(scores)
            all_masks.append(valid_top)
            audit.append([{"candidate_count": int(valid.sum()), "excluded_same_sequence": exclude_same_sequence, "excluded_scene": exclude_scene, "excluded_sample_id": bool(sample_id)}])
        return {"retrieved_ids": torch.stack(all_ids), "retrieval_scores": torch.stack(all_scores), "candidate_valid_mask": torch.stack(all_masks), "filter_audit": audit}

    def gather_future(self, ids: Tensor, query_history: Tensor) -> Tensor:
        """Return aligned [B,K,T,J,3], padding invalid IDs with zeros."""
        B, K = ids.shape
        T, J = self.items[0].future_joints.shape[:2]
        out = query_history.new_zeros(B, K, T, J, 3)
        for b in range(B):
            for k in range(K):
                idx = int(ids[b, k].item())
                if idx >= 0:
                    item = self.items[idx]
                    out[b, k] = align_future_to_query(item.future_joints.to(out), item.history_joints.to(out), query_history[b])
        return out


class MultiModalRetriever(nn.Module):
    def __init__(self, motion_dim: int = 38, joint_count: int = 23, embedding_dim: int = 256,
                 hidden_dim: int = 128, use_motion: bool = True, use_gaze: bool = True, use_scene: bool = True,
                 dropout: float = 0.1, temporal: bool = False, rich_summary: bool = False,
                 scene_focus: bool = False, gaze_heading: bool = False, scene_hist: bool = False):
        super().__init__()
        if not (use_motion or use_gaze or use_scene):
            raise ValueError("at least one retrieval modality is required")
        self.motion_dim, self.joint_count, self.embedding_dim = motion_dim, joint_count, embedding_dim
        self.use_motion, self.use_gaze, self.use_scene = use_motion, use_gaze, use_scene
        if temporal and rich_summary:
            raise ValueError('Use either temporal tokens or rich observed summaries')
        if scene_hist and not scene_focus:
            raise ValueError('Scene histogram requires gaze-centered scene features')
        self.temporal, self.rich_summary, self.scene_focus, self.gaze_heading, self.scene_hist = temporal, rich_summary, scene_focus, gaze_heading, scene_hist
        self.encoder_version = ("multimodal-retriever-v6" if scene_hist else "multimodal-retriever-v4" if gaze_heading else "multimodal-retriever-v5") if scene_focus else "multimodal-retriever-v3" if rich_summary else ENCODER_VERSION if temporal else "multimodal-retriever-v1"
        multiplier = 3 if rich_summary else 1
        self.motion_branch = nn.Sequential(nn.Linear((joint_count * 6 + motion_dim) * multiplier, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.gaze_branch = nn.Sequential(nn.Linear(7 * multiplier, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.scene_branch = nn.Sequential(nn.Linear((12 + 48) if scene_hist else 12 if scene_focus else 6, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.modality = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 2, dropout, batch_first=True, norm_first=True), num_layers=1)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, embedding_dim))
        if temporal:
            self.time_embedding = nn.Parameter(torch.randn(1, 6, hidden_dim) * 0.02)

    def forward(self, history_motion: Tensor, history_joints: Tensor, history_gaze: Tensor,
                scene: Tensor, gaze_valid: Optional[Tensor] = None) -> Tensor:
        return self.encode_features(*self.prepare_features(history_motion, history_joints, history_gaze, scene, gaze_valid))

    def prepare_features(self, history_motion: Tensor, history_joints: Tensor, history_gaze: Tensor,
                         scene: Tensor, gaze_valid: Optional[Tensor] = None):
        gaze = _as_gaze(history_gaze)
        if scene.ndim != 3 or scene.shape[-1] != 3 or (self.use_motion and (history_motion is None or history_motion.ndim != 3)):
            raise ValueError("retriever inputs must be motion[B,T,D], joints[B,T,J,3], scene[B,N,3]")
        B, T = gaze.shape[:2]
        if scene.shape[0] != B or (not self.scene_focus and (history_joints is None or history_joints.ndim != 4 or history_joints.shape[:2] != (B, T))):
            raise ValueError("retriever input batch/time mismatch")
        if self.use_motion and history_motion.shape[:2] != (B, T):
            raise ValueError("history motion/joints shape mismatch")
        if not self.scene_focus and history_joints.shape[2] != self.joint_count:
            raise ValueError("history motion/joints shape mismatch")
        valid = _normalise_mask(gaze_valid, (B, T), gaze.device)
        if self.scene_focus:
            # Endpoint query keys use gaze and scene only; observed joints remain
            # available for pose alignment after retrieval, never for this key.
            origins = gaze[:, -1]
            if self.gaze_heading:
                travel = gaze[:, -1] - gaze[:, 0]
                horizontal = torch.stack((travel[:, 0], torch.zeros_like(travel[:, 0]), travel[:, 2]), -1)
                fallback = gaze.new_tensor([0., 0., 1.]).expand_as(horizontal)
                forward = F.normalize(torch.where(horizontal.norm(dim=-1, keepdim=True) > 1e-5,
                                                  horizontal, fallback), dim=-1)
                up = gaze.new_tensor([0., 1., 0.]).expand_as(forward)
                right = torch.linalg.cross(up, forward, dim=-1)
                bases = torch.stack((right, up, forward), -1)
            else:
                bases = torch.eye(3, device=gaze.device, dtype=gaze.dtype).expand(B, -1, -1)
            joints = gaze.new_zeros(B, T, self.joint_count, 3)
        else:
            # Legacy keys keep their last-observed-pelvis reference frame.
            refs = [reference_frame(history_joints[b]) for b in range(B)]
            origins = torch.stack([r[0] for r in refs])
            bases = torch.stack([r[1] for r in refs])
            joints = torch.stack([to_reference(history_joints[b], origins[b], bases[b]) for b in range(B)])
        gaze = torch.stack([to_reference(gaze[b], origins[b], bases[b]) for b in range(B)])
        scene_local = torch.stack([to_reference(scene[b], origins[b], bases[b]) for b in range(B)])
        pelvis = joints[:, -1:, 0:1]
        if self.use_motion:
            raw_motion = _finite(history_motion)
            if raw_motion.shape[-1] >= 6:
                world_rot = axis_angle_to_matrix(raw_motion[..., :3])
                local_rot = torch.matmul(bases.transpose(-1, -2)[:, None], world_rot)
                local_aa = matrix_to_axis_angle(local_rot)
                local_trans = torch.stack([to_reference(raw_motion[b, :, 3:6], origins[b], bases[b]) for b in range(B)])
                motion_local = torch.cat([local_aa, local_trans, raw_motion[..., 6:]], dim=-1)
            else:
                motion_local = raw_motion
            joints = _finite(joints - pelvis)
            velocity = torch.diff(joints, dim=1, prepend=joints[:, :1])
            motion_feat = torch.cat([joints.flatten(2), velocity.flatten(2), motion_local], dim=-1)
        else:
            motion_feat = gaze.new_zeros(B, T, self.joint_count * 6 + self.motion_dim)
        gaze_delta = torch.diff(gaze, dim=1, prepend=gaze[:, :1])
        gaze_feat = torch.cat([gaze - pelvis.squeeze(2), gaze_delta, valid.unsqueeze(-1).float()], dim=-1)
        scene_local = _finite(scene_local - pelvis.squeeze(1))
        scene_feat = torch.cat([scene_local, scene_local.square(),], dim=-1).mean(dim=1)
        if self.scene_focus:
            # A soft neighborhood around the final gaze target retains local scene affordance.
            offset = scene_local - (gaze[:, -1] - pelvis[:, -1, 0])[:, None]
            weights = torch.softmax(-offset.square().sum(-1) / (2 * .5 ** 2), dim=1)
            focused = (torch.cat([offset, offset.square()], -1) * weights[..., None]).sum(1)
            scene_feat = torch.cat([scene_feat, focused], -1)
            if self.scene_hist:
                x = ((offset[..., 0] + 3.) / 1.5).floor().long()
                y = ((offset[..., 1] + 1.5) / 1.).floor().long()
                z = ((offset[..., 2] + 3.) / 1.5).floor().long()
                inside = (x >= 0) & (x < 4) & (y >= 0) & (y < 3) & (z >= 0) & (z < 4)
                index = (x.clamp(0, 3) * 12 + y.clamp(0, 2) * 4 + z.clamp(0, 3))
                histogram = scene_local.new_zeros(B, 48)
                histogram.scatter_add_(1, index, inside.to(scene_local.dtype))
                scene_feat = torch.cat([scene_feat, 10 * histogram / scene_local.shape[1]], -1)
        return motion_feat, gaze_feat, scene_feat

    def encode_features(self, motion_feat, gaze_feat, scene_feat):
        T = motion_feat.shape[1]
        tokens = []
        for enabled, feat, branch, idx in ((self.use_motion, motion_feat, self.motion_branch, 0), (self.use_gaze, gaze_feat, self.gaze_branch, 1), (self.use_scene, scene_feat, self.scene_branch, 2)):
            if not enabled:
                continue
            if self.temporal:
                token = branch(feat) + self.modality[idx]
                if token.ndim == 3:
                    if T > self.time_embedding.shape[1]:
                        raise ValueError("retriever history exceeds temporal embedding length")
                    token = token + self.time_embedding[:, :T]
                else:
                    token = token[:, None]
                tokens.append(token)
            else:
                if self.rich_summary and feat.ndim == 3:
                    pooled = torch.cat([feat.mean(1), feat[:, -1], feat[:, -1] - feat[:, 0]], -1)
                else:
                    pooled = feat.mean(1) if feat.ndim == 3 else feat
                tokens.append((branch(pooled) + self.modality[idx])[:, None])
        fused = self.fusion(torch.cat(tokens, dim=1)).mean(dim=1)
        return F.normalize(_finite(self.out(fused)), dim=-1, eps=1e-6)


class PrototypeEncoder(nn.Module):
    def __init__(self, joint_count: int = 23, embedding_dim: int = 256):
        super().__init__()
        self.joint_count, self.embedding_dim = joint_count, embedding_dim
        self.proj = nn.Sequential(nn.Linear(joint_count * 6, embedding_dim), nn.GELU(), nn.LayerNorm(embedding_dim))

    def forward(self, future_joints: Tensor) -> Tensor:
        if future_joints.ndim != 5 or future_joints.shape[-2:] != (self.joint_count, 3):
            raise ValueError("future_joints must be [B,K,T,J,3]")
        rel = future_joints - future_joints[:, :, :, :1]
        vel = torch.diff(rel, dim=2, prepend=rel[:, :, :1])
        return self.proj(torch.cat([rel, vel], dim=-1).flatten(-2))


class PrototypeConditionAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 256, prototype_dim: int = 256, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.prototype_proj = nn.Linear(prototype_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Output projection is zero, so the baseline is unchanged at init;
        # a small non-zero gate keeps both the projection and gate trainable.
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, future_hidden: Tensor, prototype_tokens: Tensor, candidate_valid_mask: Tensor,
                start: int = 0) -> Tuple[Tensor, Tensor]:
        if future_hidden.ndim != 3 or prototype_tokens.ndim != 4:
            raise ValueError("adapter expects hidden[B,T,D], prototypes[B,K,T,E]")
        B, L, D = future_hidden.shape
        if start < 0 or start + L > prototype_tokens.shape[2]:
            raise ValueError(f"prototype time slice [{start}, {start + L}) exceeds {prototype_tokens.shape[2]}")
        proto = self.prototype_proj(prototype_tokens[:, :, start:start + L]).flatten(1, 2)
        valid = candidate_valid_mask.bool().unsqueeze(-1).expand(-1, -1, L).flatten(1)
        # MultiheadAttention cannot represent an all-padding row.  A zero key
        # is a finite no-prototype fallback and preserves the residual contract.
        all_invalid = ~valid.any(dim=1)
        valid = valid.clone()
        valid[all_invalid, 0] = True
        proto = proto.masked_fill(~valid.unsqueeze(-1), 0.0)
        memory_mask = ~valid
        attended, _ = self.attn(future_hidden, proto, proto, key_padding_mask=memory_mask)
        residual = self.out_proj(attended)
        residual = residual.masked_fill(all_invalid[:, None, None], 0.0)
        return future_hidden + torch.tanh(self.gate) * residual, torch.sigmoid(self.gate).expand(B)


class MotionRefiner(nn.Module):
    def __init__(self, motion_dim: int = 38, joint_count: int = 23, context_dim: int = 256, prototype_dim: int = 256, hidden_dim: int = 256, pelvis_index: int = 0):
        super().__init__()
        self.motion_dim, self.joint_count, self.pelvis_index = motion_dim, joint_count, pelvis_index
        self.proto_proj = nn.Linear(prototype_dim, context_dim)
        self.context_proj = nn.Linear(context_dim, context_dim)
        self.in_proj = nn.Linear(joint_count * 3 + motion_dim + context_dim + context_dim + 2, hidden_dim)
        self.temporal = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 2, batch_first=True), num_layers=1)
        self.delta = nn.Linear(hidden_dim, joint_count * 3)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, history_motion: Tensor, base_future: Tensor, prototype_tokens: Optional[Tensor] = None,
                context: Optional[Tensor] = None, scores: Optional[Tensor] = None,
                candidate_valid_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        B, T, J, _ = base_future.shape
        if history_motion.ndim != 3 or history_motion.shape[0] != B or history_motion.shape[-1] != self.motion_dim:
            raise ValueError(f"history_motion must be [B,T,{self.motion_dim}]")
        if J != self.joint_count:
            raise ValueError("base_future joint count mismatch")
        valid = candidate_valid_mask.bool() if candidate_valid_mask is not None else None
        if scores is None:
            weights = None if valid is None else valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
            score_feat = base_future.new_zeros(B, T, 2)
        else:
            valid = torch.isfinite(scores) if valid is None else valid & torch.isfinite(scores)
            masked = scores.masked_fill(~valid, -torch.inf)
            weights = torch.nan_to_num(torch.softmax(masked, dim=-1), nan=0.0)
            top = torch.topk(masked, min(2, masked.shape[-1]), dim=-1).values
            margin = top[:, 0] - top[:, 1] if top.shape[-1] == 2 else top[:, 0]
            margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
            score_feat = torch.stack([valid.any(-1).float(), margin], dim=-1)[:, None].expand(-1, T, -1)
        if prototype_tokens is None:
            proto = base_future.new_zeros(B, T, self.context_proj.in_features)
        else:
            projected = self.proto_proj(prototype_tokens)
            proto = projected.mean(dim=1) if weights is None else (projected * weights[:, :, None, None]).sum(1)
        if context is None:
            context = base_future.new_zeros(B, self.context_proj.in_features)
        context = self.context_proj(context).unsqueeze(1).expand(-1, T, -1)
        hist = history_motion[:, -1:].expand(-1, T, -1)
        x = torch.cat([base_future.flatten(2), hist, proto, context, score_feat], dim=-1)
        delta = self.delta(self.temporal(self.in_proj(x))).view(B, T, J, 3)
        delta = delta - delta[:, :, self.pelvis_index:self.pelvis_index + 1]
        refined = base_future + torch.tanh(self.gate) * delta
        return refined, torch.sigmoid(self.gate).expand(B), delta


class PrototypeBlendRefiner(nn.Module):
    """Learn one trust weight for the retrieved pose prior while preserving root motion."""
    def __init__(self, **kwargs):
        super().__init__()
        self.mix = nn.Parameter(torch.tensor(0.0))

    def forward(self, history_motion, base_future, prototype_tokens=None, context=None,
                scores=None, candidate_valid_mask=None, prototype_future=None):
        if prototype_future is None or candidate_valid_mask is None:
            raise ValueError('Prototype blending requires aligned future candidates and a validity mask')
        valid = candidate_valid_mask.bool()
        weights = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
        relative = prototype_future - prototype_future[:, :, :, :1]
        prior = (relative * weights[:, :, None, None, None]).sum(1)
        base_relative = base_future - base_future[:, :, :1]
        delta = (prior - base_relative).masked_fill(~valid.any(-1)[:, None, None, None], 0.0)
        weight = self.mix.clamp(0.0, 0.5)
        return base_future + weight * delta, weight.expand(len(base_future)), delta


class TerminalBlendRefiner(PrototypeBlendRefiner):
    """Learn joint-specific terminal trust in retrieved futures with a gradual ramp."""
    def __init__(self, endpoint_frames=4, **kwargs):
        super().__init__()
        if not 1 <= endpoint_frames <= 10:
            raise ValueError('Terminal blend requires 1..10 terminal frames')
        self.endpoint_frames = endpoint_frames
        self.mix.data.fill_(0.30733645)
        self.mix.requires_grad_(False)
        self.terminal_mix = nn.Parameter(torch.full((1, 23, 1), 0.30733645))

    def forward(self, history_motion, base_future, prototype_tokens=None, context=None,
                scores=None, candidate_valid_mask=None, prototype_future=None):
        forecast, _, _ = super().forward(history_motion, base_future, prototype_tokens, context,
                                         scores, candidate_valid_mask, prototype_future)
        valid = candidate_valid_mask.bool()
        weights = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
        prior = ((prototype_future - prototype_future[:, :, :, :1]) * weights[:, :, None, None, None]).sum(1)
        relative = base_future - base_future[:, :, :1]
        frames = min(self.endpoint_frames, base_future.shape[1])
        ramp = base_future.new_zeros(base_future.shape[1])
        ramp[-frames:] = torch.linspace(1 / frames, 1, frames, device=base_future.device, dtype=base_future.dtype)
        correction = (self.terminal_mix.clamp(0., .8) - self.mix.clamp(0., .5))[:, None] * (prior - relative)
        correction = correction * ramp[None, :, None, None]
        correction = correction - correction[:, :, :1]
        correction = correction.masked_fill(~valid.any(-1)[:, None, None, None], 0.)
        return forecast + correction, self.terminal_mix.mean().expand(len(forecast)), correction


class AdaptiveTerminalRefiner(PrototypeBlendRefiner):
    """Observed/predicted-context trust in terminal training-pose prototypes."""
    def __init__(self, endpoint_frames=4, hidden_dim=0, joint_trust=False, part_trust=False, max_trust=.8, **kwargs):
        super().__init__()
        self.endpoint_frames = endpoint_frames
        if not 1 <= endpoint_frames <= 10:
            raise ValueError('Adaptive terminal refiner requires 1..10 terminal frames')
        self.mix.data.fill_(0.30733645); self.mix.requires_grad_(False)
        if not 0.30733645 < max_trust <= 1.5:
            raise ValueError('Terminal trust limit must exceed the initial blend and not exceed 1.5')
        if max_trust != .8:
            self.register_buffer('trust_limit', torch.tensor(float(max_trust)))
        dim = 6 + 23 * 3 * 2
        if joint_trust and part_trust:
            raise ValueError('Choose joint or body-part terminal trust, not both')
        outputs = 5 if part_trust else 23 if joint_trust else 1
        if part_trust:
            self.register_buffer('part_index', torch.tensor([0,1,2,0,1,2,0,1,2,0,1,2,0,3,4,0,3,4,3,4,3,4,0]))
        self.trust = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(.2),
                                   nn.Linear(hidden_dim, outputs)) if hidden_dim else nn.Sequential(nn.Linear(dim, outputs))
        nn.init.zeros_(self.trust[-1].weight)
        self.trust[-1].bias.data.fill_(float(torch.logit(torch.tensor(0.30733645 / max_trust))))
        self.register_buffer('feature_mean', torch.zeros(dim))
        self.register_buffer('feature_std', torch.ones(dim))

    def observation_features(self, base_future, prototype_future, valid, history_joints):
        if history_joints is None:
            raise ValueError('Adaptive terminal trust requires observed history joints')
        weights = valid.bool().float() / valid.bool().sum(-1, keepdim=True).clamp_min(1)
        relative = base_future - base_future[:, :, :1]
        candidates = prototype_future - prototype_future[:, :, :, :1]
        prior = (candidates * weights[:, :, None, None, None]).sum(1)
        gap = (prior - relative).norm(dim=-1)
        dispersion = ((candidates - prior[:, None]).norm(dim=-1).mean((-1, -2)) * weights).sum(-1)
        history = history_joints - history_joints[:, :, :1]
        history_gap = (relative[:, -1] - history[:, -1]).norm(dim=-1).mean(-1)
        summary = torch.stack([gap.mean((1, 2)), gap[:, -1].mean(1), dispersion, history_gap,
                               history[:, -1, 15, 1], relative[:, -1, 15, 1]], -1)
        return torch.cat([summary, relative[:, -1].flatten(1), prior[:, -1].flatten(1)], -1)

    def forward(self, history_motion, base_future, prototype_tokens=None, context=None,
                scores=None, candidate_valid_mask=None, prototype_future=None, history_joints=None):
        features = self.observation_features(base_future, prototype_future, candidate_valid_mask, history_joints)
        limit = self.trust_limit if hasattr(self, 'trust_limit') else .8
        trust = limit * self.trust((features - self.feature_mean) / self.feature_std.clamp_min(.05)).sigmoid()
        if hasattr(self, 'part_index'):
            trust = trust[:, self.part_index]
        weights = candidate_valid_mask.bool().float() / candidate_valid_mask.bool().sum(-1, keepdim=True).clamp_min(1)
        prior = ((prototype_future - prototype_future[:, :, :, :1]) * weights[:, :, None, None, None]).sum(1)
        relative = base_future - base_future[:, :, :1]
        frames = min(self.endpoint_frames, base_future.shape[1]); ramp = base_future.new_zeros(base_future.shape[1])
        ramp[-frames:] = torch.linspace(1 / frames, 1, frames, device=base_future.device, dtype=base_future.dtype)
        mix = self.mix.clamp(0., .5) + (trust[:, None, :, None] - self.mix.clamp(0., .5)) * ramp[None, :, None, None]
        delta = mix * (prior - relative)
        delta = delta - delta[:, :, :1]
        delta = delta.masked_fill(~candidate_valid_mask.bool().any(-1)[:, None, None, None], 0.)
        return base_future + delta, trust.mean(-1), delta


def endpoint_candidates(bank, history_joints, metadata, device=None):
    """All train endpoints aligned from histories, with query recordings excluded."""
    if metadata is None or len(metadata) != len(history_joints):
        raise ValueError('Endpoint retrieval requires recording metadata')
    cache_key = (history_joints.device.type, history_joints.device.index, history_joints.dtype)
    cache = getattr(bank, '_canonical_endpoint_cache', {})
    if cache_key not in cache:
        canonical = torch.stack([(item.future_joints[-1] - item.transform_origin) @ item.transform_basis
                                 for item in bank.items]).to(history_joints)
        cache[cache_key] = canonical
        bank._canonical_endpoint_cache = cache
    origins, bases, legal = [], [], []
    for history, row in zip(history_joints, metadata):
        if not row.get('sequence_id'):
            raise ValueError('Endpoint retrieval requires sequence_id')
        origin, basis = reference_frame(history)
        origins.append(origin); bases.append(basis)
        legal.append(torch.tensor([item.sequence_id != row['sequence_id'] for item in bank.items],
                                  device=history.device, dtype=torch.bool))
    aligned = torch.einsum('njc,bdc->bnjd', cache[cache_key], torch.stack(bases))
    aligned = aligned + torch.stack(origins)[:, None, None]
    return aligned, torch.stack(legal)


class EndpointPrototypeRefiner(PrototypeBlendRefiner):
    """Add train-bank endpoint pose guidance gradually to the terminal frames."""
    def __init__(self, endpoint_topk=3, endpoint_frames=4, **kwargs):
        super().__init__()
        if endpoint_topk < 1 or not 1 <= endpoint_frames <= 10:
            raise ValueError('Invalid endpoint retrieval configuration')
        self.endpoint_topk, self.endpoint_frames = endpoint_topk, endpoint_frames
        self.mix.data.fill_(0.30733645)
        self.mix.requires_grad_(False)
        self.endpoint_mix = nn.Parameter(torch.zeros(1, 23, 1))

    def forward(self, history_motion, base_future, prototype_tokens=None, context=None,
                scores=None, candidate_valid_mask=None, prototype_future=None,
                endpoint_future=None, endpoint_valid=None):
        forecast, _, _ = super().forward(history_motion, base_future, prototype_tokens, context,
                                         scores, candidate_valid_mask, prototype_future)
        if endpoint_future is None or endpoint_valid is None:
            raise ValueError('Endpoint refinement requires train endpoint candidates')
        relative = endpoint_future - endpoint_future[:, :, :1]
        predicted_endpoint = forecast[:, -1] - forecast[:, -1, :1]
        distances = (relative - predicted_endpoint[:, None]).norm(dim=-1).mean(-1)
        distances = distances.masked_fill(~endpoint_valid.bool(), torch.inf)
        k = min(self.endpoint_topk, relative.shape[1])
        values, ids = distances.topk(k, largest=False)
        selected = relative.gather(1, ids[:, :, None, None].expand(-1, -1, 23, 3))
        valid = torch.isfinite(values)
        weights = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
        prior = (selected * weights[:, :, None, None]).sum(1)
        delta = (prior - predicted_endpoint) * self.endpoint_mix.clamp(0.0, 0.8)
        delta = delta - delta[:, :1]
        delta = delta.masked_fill(~valid.any(-1)[:, None, None], 0.0)
        frames = min(self.endpoint_frames, forecast.shape[1])
        ramp = forecast.new_zeros(forecast.shape[1])
        ramp[-frames:] = torch.linspace(1 / frames, 1, frames, device=forecast.device, dtype=forecast.dtype)
        correction = delta[:, None] * ramp[None, :, None, None]
        return forecast + correction, self.endpoint_mix.mean().expand(len(forecast)), correction


class LocalMotionRefiner(nn.Module):
    """Refine in a history-defined frame with confidence-weighted, time-indexed prototypes."""
    def __init__(self, motion_dim=38, joint_count=23, context_dim=256, prototype_dim=256,
                 hidden_dim=128, pelvis_index=0, max_delta=0.08):
        super().__init__()
        self.max_delta = max_delta
        self.core = MotionRefiner(motion_dim, joint_count, context_dim, prototype_dim,
                                  hidden_dim, pelvis_index)
        self.time = nn.Parameter(torch.randn(1, 10, hidden_dim) * 0.02)
        # Use one small temporal layer with dropout; no new world-frame parameters.

    def forward(self, history_motion, base_future, prototype_tokens=None, context=None,
                scores=None, candidate_valid_mask=None, history_joints=None):
        if history_joints is None:
            raise ValueError("local refiner requires observed history_joints")
        refs = [reference_frame(row) for row in history_joints]
        basis = torch.stack([row[1] for row in refs])
        origin = torch.stack([row[0] for row in refs])
        local_base = torch.stack([to_reference(row, o, b) for row, o, b in zip(base_future, origin, basis)])
        local_base = local_base - local_base[:, :, :1]
        rotation = axis_angle_to_matrix(history_motion[..., :3])
        local_ori = matrix_to_axis_angle(basis.transpose(-1, -2)[:, None] @ rotation)
        local_trans = torch.stack([to_reference(row[:, 3:6], o, b) for row, o, b in zip(history_motion, origin, basis)])
        local_motion = torch.cat([local_ori, local_trans, history_motion[..., 6:]], dim=-1)
        # Keep frame order visible to the temporal residual network.
        core = self.core
        B, T, J, _ = local_base.shape
        valid = torch.ones((B, 1), device=base_future.device, dtype=torch.bool) if candidate_valid_mask is None else candidate_valid_mask.bool()
        if prototype_tokens is None:
            proto = local_base.new_zeros(B, T, core.context_proj.in_features)
        else:
            if scores is None:
                weights = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
            else:
                weights = torch.nan_to_num(torch.softmax(scores.masked_fill(~valid, -torch.inf) / 0.1, -1), nan=0.0)
            proto = (core.proto_proj(prototype_tokens) * weights[:, :, None, None]).sum(1)
        if context is None:
            context = local_base.new_zeros(B, core.context_proj.in_features)
        ctx = core.context_proj(context)[:, None].expand(-1, T, -1)
        score_feat = torch.stack([valid.any(-1).float(), valid.float().mean(-1)], -1)[:, None].expand(-1, T, -1)
        inputs = torch.cat([local_base.flatten(2), local_motion[:, -1:].expand(-1, T, -1), proto, ctx, score_feat], -1)
        hidden = core.temporal(core.in_proj(inputs) + self.time[:, :T])
        local_delta = self.max_delta * torch.tanh(core.delta(hidden).view(B, T, J, 3))
        local_delta = local_delta - local_delta[:, :, :1]
        delta = torch.matmul(local_delta, basis.transpose(-1, -2)[:, None])
        refined = base_future + torch.tanh(core.gate) * delta
        return refined, torch.sigmoid(core.gate).expand(B), delta


def retrieval_soft_label_loss(similarity: Tensor, future_distance: Tensor, temperature: float = 0.1) -> Tensor:
    """Differentiable soft target loss for sampled legal candidates."""
    target = F.softmax(-future_distance / temperature, dim=-1)
    pred = F.log_softmax(similarity / temperature, dim=-1)
    return F.kl_div(pred, target, reduction="batchmean")


def bank_manifest_hash(items: Iterable[MotionBankItem]) -> str:
    payload = json.dumps([i.metadata() for i in items], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def forecast_candidate_scores(base_future, prototype_future, valid, local=False):
    """Use a model forecast to weight candidates; this function has no label argument."""
    base_relative = base_future - base_future[:, :, :1]
    prototype_relative = prototype_future - prototype_future[:, :, :, :1]
    distance = (prototype_relative - base_relative[:, None]).norm(dim=-1).mean(dim=(-1, -2))
    scores = -distance / 0.05
    # LocalMotionRefiner applies its own 0.1 softmax temperature.
    return (scores * (0.1 if local else 1.0)).masked_fill(~valid.bool(), -torch.inf)


def retriever_from_bank(bank, dropout=0.0):
    model = MultiModalRetriever(embedding_dim=int(bank.manifest.get('embedding_dim', 256)),
                               use_motion=bank.manifest.get('use_motion', True),
                               use_gaze=bank.manifest.get('use_gaze', True),
                               use_scene=bank.manifest.get('use_scene', True), dropout=dropout,
                               temporal=bank.manifest.get('encoder_version') == ENCODER_VERSION,
                               rich_summary=bank.manifest.get('encoder_version') == 'multimodal-retriever-v3',
                               scene_focus=bank.manifest.get('encoder_version') in ('multimodal-retriever-v4', 'multimodal-retriever-v5', 'multimodal-retriever-v6'),
                               gaze_heading=bank.manifest.get('encoder_version') == 'multimodal-retriever-v4',
                               scene_hist=bank.manifest.get('encoder_version') == 'multimodal-retriever-v6')
    if bank.encoder_state_dict is None:
        raise ValueError('Bank has no embedded encoder')
    model.load_state_dict(bank.encoder_state_dict, strict=True)
    return model
