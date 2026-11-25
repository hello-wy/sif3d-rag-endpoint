from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


YELLOW_HEX = "#E9C911"
BLUE_HEX = "#2F68CF"
Y_UP_AXIS = 1
RADIAL_AXES = (0, 2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render lab0220 per-step cylindrical overlays using GT pelvis positions."
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("/data/wuyang/human/GIMO/lab0220/scene_obj"),
        help="Path to the scene_obj directory.",
    )
    parser.add_argument(
        "--pelvis-file",
        type=Path,
        default=Path("/data/wuyang/output/visual/lab0220_21/pelvis_global_positions.txt"),
        help="Pelvis position export produced by the evaluation pipeline.",
    )
    parser.add_argument(
        "--step-prefix",
        type=str,
        default="GT",
        choices=("GT", "PD"),
        help="Which future sequence to visualize.",
    )
    parser.add_argument(
        "--history-prefix",
        type=str,
        default="IP",
        help="Prefix used to locate the last input-step pelvis for step-0 radius estimation.",
    )
    parser.add_argument(
        "--expected-steps",
        type=int,
        default=10,
        help="How many future steps to render.",
    )
    parser.add_argument(
        "--base-radius",
        type=float,
        default=0.7,
        help="Base cylinder radius copied from 切割方法.py.",
    )
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=2.0,
        help="Radius expansion factor copied from 切割方法.py.",
    )
    parser.add_argument(
        "--cylinder-half-height",
        type=float,
        default=2.0,
        help="Cylinder half height in world-space meters.",
    )
    parser.add_argument(
        "--height-threshold",
        type=float,
        default=0.5,
        help="World-space Y threshold for two-tone coloring.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.92,
        help="Transparency used when overlaying the colored cut mesh on the original scene.",
    )
    parser.add_argument(
        "--overlay-lift",
        type=float,
        default=0.006,
        help="Small normal-direction offset to reduce z-fighting against the base scene.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1600,
        help="Output image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1200,
        help="Output image height.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=52.0,
        help="Perspective camera vertical field of view in degrees.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lab0220_gt_cylinder_overlay"),
        help="Directory used for rendered frames and metadata.",
    )
    return parser.parse_args()


def hex_to_rgb01(value: str) -> np.ndarray:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got: {value}")
    rgb = np.array(
        [int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)],
        dtype=np.float64,
    )
    return rgb / 255.0


def distance_3d(point_a: np.ndarray, point_b: np.ndarray) -> float:
    delta = np.asarray(point_a, dtype=np.float64) - np.asarray(point_b, dtype=np.float64)
    return float(np.linalg.norm(delta))


def compute_dynamic_radius(step_distance: float, base_radius: float, lambda_scale: float) -> float:
    return float(base_radius) + float(lambda_scale) * float(step_distance)


def parse_pelvis_positions(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            name, coords_text = line.split(":", 1)
            coords = np.fromstring(coords_text.strip().strip("[]"), sep=",", dtype=np.float64)
            if coords.shape != (3,):
                raise ValueError(f"Invalid pelvis line: {line}")
            result[name.strip()] = coords
    if not result:
        raise ValueError(f"No pelvis positions found in {path}")
    return result


def collect_ordered_centers(
    pelvis_positions: dict[str, np.ndarray],
    prefix: str,
    expected_steps: int,
) -> list[np.ndarray]:
    centers: list[np.ndarray] = []
    for step_idx in range(expected_steps):
        key = f"{prefix}_{step_idx}"
        if key not in pelvis_positions:
            raise KeyError(f"Missing pelvis position: {key}")
        centers.append(np.asarray(pelvis_positions[key], dtype=np.float64))
    return centers


def resolve_prev_center(pelvis_positions: dict[str, np.ndarray], history_prefix: str) -> np.ndarray | None:
    candidates = []
    for key, value in pelvis_positions.items():
        if not key.startswith(f"{history_prefix}_"):
            continue
        try:
            step_idx = int(key.split("_")[-1])
        except ValueError:
            continue
        candidates.append((step_idx, value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return np.asarray(candidates[-1][1], dtype=np.float64)


def build_step_specs(
    centers: list[np.ndarray],
    prev_center: np.ndarray | None,
    base_radius: float,
    lambda_scale: float,
    half_height: float,
) -> list[dict]:
    specs = []
    for index, center in enumerate(centers):
        if index == 0 and prev_center is not None:
            step_distance = distance_3d(center, prev_center)
        elif index == 0 and len(centers) > 1:
            step_distance = distance_3d(centers[1], center)
        else:
            step_distance = distance_3d(center, centers[index - 1])
        specs.append(
            {
                "index": index,
                "center": center.astype(np.float64),
                "step_distance": float(step_distance),
                "radius": compute_dynamic_radius(step_distance, base_radius, lambda_scale),
                "half_height": float(half_height),
            }
        )
    return specs


def load_scene_mesh(scene_dir: Path) -> tuple[o3d.geometry.TriangleMesh, trimesh.Trimesh, np.ndarray]:
    obj_path = scene_dir / "textured_output.obj"
    if not obj_path.exists():
        raise FileNotFoundError(f"Scene OBJ not found: {obj_path}")

    scene_o3d = o3d.io.read_triangle_mesh(str(obj_path), True)
    if scene_o3d.is_empty():
        raise RuntimeError(f"Open3D failed to load scene mesh: {obj_path}")
    scene_o3d.compute_vertex_normals()

    scene_tm = trimesh.load(str(obj_path), force="mesh", process=False)
    if scene_tm.is_empty:
        raise RuntimeError(f"trimesh failed to load scene mesh: {obj_path}")

    vertices = np.asarray(scene_tm.vertices, dtype=np.float64)
    faces = np.asarray(scene_tm.faces, dtype=np.int64)
    face_centers = vertices[faces].mean(axis=1)

    return scene_o3d, scene_tm, face_centers


def extract_cylinder_submesh(
    scene_mesh: trimesh.Trimesh,
    face_centers: np.ndarray,
    center: np.ndarray,
    radius: float,
    half_height: float,
) -> trimesh.Trimesh:
    radial_delta = face_centers[:, RADIAL_AXES] - center[list(RADIAL_AXES)][None, :]
    radial_mask = np.sum(radial_delta * radial_delta, axis=1) <= float(radius) ** 2
    height_mask = np.abs(face_centers[:, Y_UP_AXIS] - float(center[Y_UP_AXIS])) <= float(half_height)
    face_mask = radial_mask & height_mask

    if not np.any(face_mask):
        raise ValueError(
            f"No geometry found in cylinder cut. center={center.tolist()} radius={radius:.4f} half_height={half_height:.4f}"
        )

    submesh = scene_mesh.submesh([face_mask], append=True, repair=False)
    if submesh.is_empty:
        raise ValueError("Cylinder cut produced an empty mesh after submesh extraction.")
    return submesh


def colorize_overlay_mesh(
    mesh: trimesh.Trimesh,
    height_threshold: float,
    overlay_lift: float,
) -> o3d.geometry.TriangleMesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(vertices)
    legacy.triangles = o3d.utility.Vector3iVector(faces)
    legacy.compute_vertex_normals()

    normals = np.asarray(legacy.vertex_normals)
    if overlay_lift > 0.0 and normals.shape == vertices.shape:
        vertices = vertices + normals * float(overlay_lift)
        legacy.vertices = o3d.utility.Vector3dVector(vertices)
        legacy.compute_vertex_normals()

    yellow = hex_to_rgb01(YELLOW_HEX)
    blue = hex_to_rgb01(BLUE_HEX)
    colors = np.repeat(blue[None, :], repeats=vertices.shape[0], axis=0)
    colors[vertices[:, Y_UP_AXIS] > float(height_threshold)] = yellow
    legacy.vertex_colors = o3d.utility.Vector3dVector(colors)
    return legacy


def build_camera(scene_bounds: np.ndarray, pelvis_centers: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene_min = scene_bounds[0].astype(np.float64)
    scene_max = scene_bounds[1].astype(np.float64)
    centers = np.stack(pelvis_centers, axis=0)

    path_min = centers.min(axis=0)
    path_max = centers.max(axis=0)
    combined_min = np.minimum(scene_min, path_min)
    combined_max = np.maximum(scene_max, path_max)
    combined_center = 0.5 * (combined_min + combined_max)
    extent = np.maximum(combined_max - combined_min, 1e-3)
    scale = float(np.max(extent))

    path_center = 0.5 * (path_min + path_max)
    lookat = 0.35 * combined_center + 0.65 * path_center
    lookat[Y_UP_AXIS] = max(float(path_min[Y_UP_AXIS]) + 0.25, float(path_center[Y_UP_AXIS]))

    eye = lookat + np.array([0.62 * scale, 0.54 * scale, 0.92 * scale], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return lookat.astype(np.float32), eye.astype(np.float32), up.astype(np.float32)


def write_metadata(output_dir: Path, metadata: dict):
    path = output_dir / "render_metadata.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def render_sequence(
    scene_mesh: o3d.geometry.TriangleMesh,
    step_specs: list[dict],
    overlay_meshes: list[o3d.geometry.TriangleMesh],
    args,
):
    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.scene.set_sun_light([0.45, -1.0, -0.35], [1.0, 1.0, 1.0], 85000.0)
    renderer.scene.scene.enable_sun_light(True)

    scene_material = o3d.visualization.rendering.MaterialRecord()
    scene_material.shader = "defaultLit"
    scene_material.base_color = (0.86, 0.86, 0.86, 1.0)
    renderer.scene.add_geometry("base_scene", scene_mesh, scene_material)

    overlay_material = o3d.visualization.rendering.MaterialRecord()
    overlay_material.shader = "defaultLitTransparency"
    overlay_material.base_color = (1.0, 1.0, 1.0, float(args.overlay_alpha))

    scene_bounds = scene_mesh.get_axis_aligned_bounding_box()
    centers = [spec["center"] for spec in step_specs]
    lookat, eye, up = build_camera(
        np.stack([scene_bounds.get_min_bound(), scene_bounds.get_max_bound()], axis=0),
        centers,
    )
    renderer.setup_camera(float(args.fov), lookat, eye, up)

    for spec, overlay in zip(step_specs, overlay_meshes):
        overlay_name = f"overlay_{spec['index']:02d}"
        renderer.scene.add_geometry(overlay_name, overlay, overlay_material)
        image = renderer.render_to_image()
        output_path = args.output_dir / f"{args.step_prefix.lower()}_step_{spec['index']:02d}.png"
        o3d.io.write_image(str(output_path), image)
        renderer.scene.remove_geometry(overlay_name)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pelvis_positions = parse_pelvis_positions(args.pelvis_file)
    future_centers = collect_ordered_centers(pelvis_positions, args.step_prefix, args.expected_steps)
    prev_center = resolve_prev_center(pelvis_positions, args.history_prefix)

    step_specs = build_step_specs(
        centers=future_centers,
        prev_center=prev_center,
        base_radius=args.base_radius,
        lambda_scale=args.lambda_scale,
        half_height=args.cylinder_half_height,
    )

    scene_o3d, scene_tm, face_centers = load_scene_mesh(args.scene_dir)
    overlay_meshes = []
    for spec in step_specs:
        cut_mesh = extract_cylinder_submesh(
            scene_mesh=scene_tm,
            face_centers=face_centers,
            center=spec["center"],
            radius=spec["radius"],
            half_height=spec["half_height"],
        )
        overlay_meshes.append(
            colorize_overlay_mesh(
                mesh=cut_mesh,
                height_threshold=args.height_threshold,
                overlay_lift=args.overlay_lift,
            )
        )

    render_sequence(
        scene_mesh=scene_o3d,
        step_specs=step_specs,
        overlay_meshes=overlay_meshes,
        args=args,
    )

    metadata = {
        "scene_dir": str(args.scene_dir),
        "pelvis_file": str(args.pelvis_file),
        "step_prefix": args.step_prefix,
        "history_prefix": args.history_prefix,
        "expected_steps": int(args.expected_steps),
        "base_radius": float(args.base_radius),
        "lambda_scale": float(args.lambda_scale),
        "cylinder_half_height": float(args.cylinder_half_height),
        "height_threshold": float(args.height_threshold),
        "overlay_alpha": float(args.overlay_alpha),
        "overlay_lift": float(args.overlay_lift),
        "camera_fov": float(args.fov),
        "steps": [
            {
                "index": int(spec["index"]),
                "center": [float(v) for v in spec["center"]],
                "step_distance": float(spec["step_distance"]),
                "radius": float(spec["radius"]),
                "half_height": float(spec["half_height"]),
                "image_path": str(args.output_dir / f"{args.step_prefix.lower()}_step_{spec['index']:02d}.png"),
            }
            for spec in step_specs
        ],
    }
    write_metadata(args.output_dir, metadata)

    print(f"Rendered {len(step_specs)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
