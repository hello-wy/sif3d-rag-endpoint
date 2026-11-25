"""
Import SMPL-X rig data exported by eval_mod.py into Blender.

Usage:
    blender --python scripts/blender_import_smplx_rig.py -- \
        --sequence-dir /path/to/export_dir \
        --prefix PD

Expected files inside --sequence-dir:
    smplx_rig_template.npz
    PD_0_rig.npz
    PD_1_rig.npz
    ...

This script reconstructs:
1. A rest mesh from `smplx_rig_template.npz`
2. An armature with the SMPL-X hierarchy
3. Vertex groups from `skinning_weights`
4. A body animation from `global_orient` + `body_pose`

Notes:
- The current exporter stores body pose for the main 22 SMPL-X body joints.
- Hand, jaw, and eye joints are created in the rig but remain in rest pose.
- Bone display is synthetic on purpose: all deform bones share a stable local basis,
  which makes SMPL axis-angle rotations easier to replay inside Blender.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import List

import bpy
import numpy as np
from mathutils import Vector


JOINT_NAMES: List[str] = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]

BODY_JOINT_COUNT = 22
TAIL_OFFSET = np.array([0.0, 0.05, 0.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--prefix", default="PD")
    parser.add_argument("--template", default=None)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--clear-scene", action="store_true")
    return parser.parse_args(argv)


def axis_angle_to_blender(value: np.ndarray) -> tuple[float, float, float, float]:
    angle = float(np.linalg.norm(value))
    if angle < 1e-8:
        return (0.0, 0.0, 1.0, 0.0)
    axis = value / angle
    return (angle, float(axis[0]), float(axis[1]), float(axis[2]))


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for armature in list(bpy.data.armatures):
        bpy.data.armatures.remove(armature)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def load_template(template_path: str) -> dict:
    with np.load(template_path) as data:
        return {key: data[key] for key in data.files}


def list_rig_files(sequence_dir: str, prefix: str) -> List[str]:
    pattern = os.path.join(sequence_dir, f"{prefix}_*_rig.npz")
    rig_files = glob.glob(pattern)

    def frame_key(path: str) -> int:
        match = re.search(rf"{re.escape(prefix)}_(\d+)_rig\.npz$", os.path.basename(path))
        if match is None:
            return -1
        return int(match.group(1))

    return sorted(rig_files, key=frame_key)


def create_mesh_object(name: str, vertices: np.ndarray, faces: np.ndarray) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_armature(
    name: str,
    joints: np.ndarray,
    parents: np.ndarray,
) -> bpy.types.Object:
    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = arm_data.edit_bones
    for joint_index, joint in enumerate(joints):
        bone = edit_bones.new(JOINT_NAMES[joint_index])
        head = Vector(joint.tolist())
        bone.head = head
        bone.tail = head + Vector(TAIL_OFFSET.tolist())

    for joint_index, parent_index in enumerate(parents[: len(joints)]):
        if parent_index < 0 or joint_index == 0:
            continue
        edit_bones[JOINT_NAMES[joint_index]].parent = edit_bones[JOINT_NAMES[int(parent_index)]]

    bpy.ops.object.mode_set(mode="OBJECT")
    arm_data.display_type = "STICK"
    return arm_obj


def bind_mesh_to_armature(
    mesh_obj: bpy.types.Object,
    arm_obj: bpy.types.Object,
    weights: np.ndarray,
) -> None:
    for joint_name in JOINT_NAMES:
        mesh_obj.vertex_groups.new(name=joint_name)

    for vertex_index in range(weights.shape[0]):
        for joint_index, weight in enumerate(weights[vertex_index, : len(JOINT_NAMES)]):
            if weight <= 1e-6:
                continue
            mesh_obj.vertex_groups[JOINT_NAMES[joint_index]].add([vertex_index], float(weight), "REPLACE")

    modifier = mesh_obj.modifiers.new(name="SMPLX_Armature", type="ARMATURE")
    modifier.object = arm_obj


def set_rest_pose(arm_obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")
    for pose_bone in arm_obj.pose.bones:
        pose_bone.rotation_mode = "AXIS_ANGLE"
        pose_bone.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)
    bpy.ops.object.mode_set(mode="OBJECT")


def apply_frame_pose(
    arm_obj: bpy.types.Object,
    rest_joints: np.ndarray,
    rig_path: str,
    frame_number: int,
) -> None:
    with np.load(rig_path) as data:
        joints = data["joints"]
        global_orient = data["global_orient"]
        body_pose = data["body_pose"].reshape(21, 3)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")

    pelvis = arm_obj.pose.bones[JOINT_NAMES[0]]
    pelvis.rotation_mode = "AXIS_ANGLE"
    pelvis.rotation_axis_angle = axis_angle_to_blender(global_orient)
    pelvis.location = tuple((joints[0] - rest_joints[0]).tolist())
    pelvis.keyframe_insert(data_path="rotation_axis_angle", frame=frame_number)
    pelvis.keyframe_insert(data_path="location", frame=frame_number)

    for joint_index in range(1, BODY_JOINT_COUNT):
        pose_bone = arm_obj.pose.bones[JOINT_NAMES[joint_index]]
        pose_bone.rotation_mode = "AXIS_ANGLE"
        pose_bone.rotation_axis_angle = axis_angle_to_blender(body_pose[joint_index - 1])
        pose_bone.keyframe_insert(data_path="rotation_axis_angle", frame=frame_number)

    bpy.ops.object.mode_set(mode="OBJECT")


def main() -> None:
    args = parse_args()
    if args.clear_scene:
        clear_scene()

    template_path = args.template or os.path.join(args.sequence_dir, "smplx_rig_template.npz")
    rig_files = list_rig_files(args.sequence_dir, args.prefix)
    if not rig_files:
        raise FileNotFoundError(f"No rig files found for prefix {args.prefix} in {args.sequence_dir}")

    template = load_template(template_path)
    rest_vertices = template["rest_vertices"]
    rest_joints = template["rest_joints"]
    faces = template["faces"]
    parents = template["joint_parents"]
    weights = template["skinning_weights"]

    mesh_obj = create_mesh_object(f"{args.prefix}_mesh", rest_vertices, faces)
    arm_obj = create_armature(f"{args.prefix}_armature", rest_joints, parents)
    bind_mesh_to_armature(mesh_obj, arm_obj, weights)
    set_rest_pose(arm_obj)

    for frame_offset, rig_path in enumerate(rig_files):
        apply_frame_pose(
            arm_obj=arm_obj,
            rest_joints=rest_joints,
            rig_path=rig_path,
            frame_number=args.start_frame + frame_offset,
        )

    bpy.context.scene.frame_start = args.start_frame
    bpy.context.scene.frame_end = args.start_frame + len(rig_files) - 1
    bpy.context.view_layer.objects.active = mesh_obj
    print(f"Imported {len(rig_files)} frames from {args.sequence_dir} with prefix {args.prefix}")


if __name__ == "__main__":
    main()
