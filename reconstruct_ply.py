#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil
import tempfile
from loguru import logger
import numpy as np
import torch

from visual_util import (
    _images_to_rgb,
    _limit_points,
    apply_sky_mask,
    depth_edge,
    predictions_to_glb,
)
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(input_dir: str) -> tuple[list[str], str]:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    image_paths: list[str] = []
    for name in os.listdir(input_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_IMAGE_EXTS:
            image_paths.append(os.path.join(input_dir, name))
    if image_paths:
        return sorted(image_paths), input_dir

    images_subdir = os.path.join(input_dir, "images")
    if os.path.isdir(images_subdir):
        for name in os.listdir(images_subdir):
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTS:
                image_paths.append(os.path.join(images_subdir, name))
    return sorted(image_paths), images_subdir


def load_model(checkpoint_path: str) -> VGGTOmega:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run VGGT-Omega.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # log model size in MB
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    logger.info(f"Loaded VGGT-Omega model from {checkpoint_path} ({model_size_mb:.2f} MB)")

    return model.to("cuda")


def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def resolve_sky_target_dir(image_dir: str, mask_sky: bool) -> tuple[str | None, str | None]:
    if not mask_sky:
        return None, None
    if os.path.basename(image_dir) == "images" and os.path.isdir(os.path.dirname(image_dir)):
        return os.path.dirname(image_dir), None
    temp_dir = tempfile.mkdtemp(prefix="vggt_omega_")
    os.symlink(os.path.abspath(image_dir), os.path.join(temp_dir, "images"))
    return temp_dir, temp_dir


def extract_points_from_predictions(
    predictions: dict,
    conf_thres: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
    mask_sky: bool,
    target_dir: str | None,
    max_points: int,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
) -> tuple[np.ndarray, np.ndarray]:
    conf_thres = max(2.0, float(conf_thres))
    points = predictions["world_points_from_depth"]
    conf = predictions["depth_conf"]
    if filter_depth_edges and "depth" in predictions:
        conf = conf.copy()
        conf[depth_edge(predictions["depth"][..., 0], rtol=depth_edge_rtol)] = 0.0
    if mask_sky and target_dir is not None:
        conf = apply_sky_mask(conf, target_dir)

    images = predictions["images"]
    vertices = points.reshape(-1, 3)
    colors = _images_to_rgb(images).reshape(-1, 3)
    colors = (colors * 255).clip(0, 255).astype(np.uint8)
    conf = conf.reshape(-1)

    mask = np.isfinite(vertices).all(axis=1) & np.isfinite(conf)
    if conf_thres > 0 and np.any(mask):
        conf_threshold = np.percentile(conf[mask], conf_thres)
        mask &= conf >= conf_threshold
    mask &= conf > 1e-5

    if mask_black_bg:
        mask &= colors.sum(axis=1) >= 16
    if mask_white_bg:
        mask &= ~((colors[:, 0] > 240) & (colors[:, 1] > 240) & (colors[:, 2] > 240))

    vertices = vertices[mask]
    colors = colors[mask]
    vertices, colors = _limit_points(vertices, colors, max_points)
    return vertices, colors


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    quat /= np.linalg.norm(quat) if np.linalg.norm(quat) > 0 else 1.0
    return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])


def build_point_cloud(
    image_paths: list[str],
    image_dir: str,
    checkpoint_path: str,
    image_resolution: int,
    resize_mode: str,
    conf_thres: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
    mask_sky: bool,
    max_points: int,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
) -> tuple[dict, "trimesh.PointCloud"]:
    logger.info("Loading VGGT-Omega model...")
    model = load_model(checkpoint_path)
    logger.info("Model loaded. Preprocessing images...")
    images = load_and_preprocess_images(
        image_paths,
        mode=resize_mode,
        image_resolution=image_resolution,
    ).to("cuda")
    logger.info(f"Loaded images with shape: {images.shape} and dtype: {images.dtype}")

    logger.info("Running inference with VGGT-Omega...")

    with torch.inference_mode():
        predictions = model(images)

    logger.info("Inference completed. Processing predictions...")

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np: dict = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
        predictions_np[key] = value

    predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )

    sky_target_dir, temp_dir = resolve_sky_target_dir(image_dir, mask_sky)

    try:
        scene = predictions_to_glb(
            predictions_np,
            conf_thres=conf_thres,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=False,
            mask_sky=mask_sky,
            target_dir=sky_target_dir,
            max_points=max_points,
            filter_depth_edges=filter_depth_edges,
            depth_edge_rtol=depth_edge_rtol,
        )
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    if len(scene.geometry) == 0:
        raise RuntimeError("No point cloud generated from predictions.")
    point_cloud = next(iter(scene.geometry.values()))
    return predictions_np, point_cloud


def export_colmap(
    predictions: dict,
    image_paths: list[str],
    image_dir: str,
    output_dir: str,
    conf_thres: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
    mask_sky: bool,
    max_points: int,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    images = predictions["images"]
    if images.ndim != 4:
        raise ValueError("Predicted images must have shape (N, C, H, W) or (N, H, W, C).")
    if images.shape[1] == 3:
        height, width = int(images.shape[2]), int(images.shape[3])
    else:
        height, width = int(images.shape[1]), int(images.shape[2])

    intrinsics = predictions["intrinsic"]
    extrinsics = predictions["extrinsic"]
    num_images = min(len(image_paths), intrinsics.shape[0], extrinsics.shape[0])
    if num_images == 0:
        raise RuntimeError("No camera predictions available for COLMAP export.")
    if len(image_paths) != num_images:
        logger.warning("Image count does not match predictions. Truncating to the shortest length.")

    cameras_path = os.path.join(output_dir, "cameras.txt")
    images_path = os.path.join(output_dir, "images.txt")
    points_path = os.path.join(output_dir, "points3D.txt")

    with open(cameras_path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(num_images))
        for idx in range(num_images):
            fx = float(intrinsics[idx, 0, 0])
            fy = float(intrinsics[idx, 1, 1])
            cx = float(intrinsics[idx, 0, 2])
            cy = float(intrinsics[idx, 1, 2])
            camera_id = idx + 1
            f.write(
                "{} PINHOLE {} {} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    camera_id, width, height, fx, fy, cx, cy
                )
            )

    with open(images_path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}, mean observations per image: 0\n".format(num_images))
        for idx in range(num_images):
            rotation = extrinsics[idx, :3, :3]
            translation = extrinsics[idx, :3, 3]
            qw, qx, qy, qz = rotation_matrix_to_quaternion(rotation)
            camera_id = idx + 1
            image_id = idx + 1
            image_name = os.path.basename(image_paths[idx])
            f.write(
                "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {} {}\n\n".format(
                    image_id,
                    qw,
                    qx,
                    qy,
                    qz,
                    float(translation[0]),
                    float(translation[1]),
                    float(translation[2]),
                    camera_id,
                    image_name,
                )
            )

    sky_target_dir, temp_dir = resolve_sky_target_dir(image_dir, mask_sky)
    try:
        vertices, colors = extract_points_from_predictions(
            predictions,
            conf_thres=conf_thres,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            mask_sky=mask_sky,
            target_dir=sky_target_dir,
            max_points=max_points,
            filter_depth_edges=filter_depth_edges,
            depth_edge_rtol=depth_edge_rtol,
        )
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    with open(points_path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: {}\n".format(len(vertices)))
        for idx, (vertex, color) in enumerate(zip(vertices, colors), start=1):
            f.write(
                "{} {:.6f} {:.6f} {:.6f} {} {} {} {:.6f}\n".format(
                    idx,
                    float(vertex[0]),
                    float(vertex[1]),
                    float(vertex[2]),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                    0.0,
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT-Omega reconstruction and export PLY or COLMAP outputs.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to VGGT-Omega checkpoint (.pt).")
    parser.add_argument("--input-dir", required=True, help="Directory containing input images.")
    parser.add_argument(
        "--output-format",
        choices=["ply", "colmap"],
        default="ply",
        help="Output format to export.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path. For ply: file path (default: <input-dir>/scene.ply). "
            "For colmap: directory (default: <input-dir>/colmap)."
        ),
    )
    parser.add_argument("--image-resolution", type=int, default=512, help="Image resolution for inference.")
    parser.add_argument(
        "--resize-mode",
        choices=["balanced", "max_size"],
        default="balanced",
        help="Resize strategy before inference.",
    )
    parser.add_argument("--conf-thres", type=float, default=20.0, help="Confidence percentile threshold.")
    parser.add_argument("--mask-black-bg", action="store_true", help="Filter near-black pixels.")
    parser.add_argument("--mask-white-bg", action="store_true", help="Filter near-white pixels.")
    parser.add_argument("--mask-sky", action="store_true", help="Filter sky region with skyseg.")
    parser.add_argument("--max-points", type=int, default=300000, help="Maximum output points.")
    parser.add_argument(
        "--no-filter-depth-edges",
        action="store_true",
        help="Disable depth edge filtering.",
    )
    parser.add_argument(
        "--depth-edge-rtol",
        type=float,
        default=0.03,
        help="Relative threshold for depth edge filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths, image_dir = collect_images(args.input_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"No supported images found in: {args.input_dir}")
    else:
        logger.info(f"Found {len(image_paths)} images for reconstruction.")


    if args.output_format == "ply":
        output_path = args.output or os.path.join(args.input_dir, "scene.ply")
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, "scene.ply")
    else:
        output_path = args.output or os.path.join(args.input_dir, "colmap")
        if os.path.isfile(output_path):
            raise RuntimeError(f"COLMAP output must be a directory, got file: {output_path}")

    predictions_np, point_cloud = build_point_cloud(
        image_paths=image_paths,
        image_dir=image_dir,
        checkpoint_path=args.checkpoint,
        image_resolution=args.image_resolution,
        resize_mode=args.resize_mode,
        conf_thres=args.conf_thres,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        mask_sky=args.mask_sky,
        max_points=args.max_points,
        filter_depth_edges=not args.no_filter_depth_edges,
        depth_edge_rtol=args.depth_edge_rtol,
    )
    if args.output_format == "ply":
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        point_cloud.export(output_path)
        logger.info(f"Reconstruction completed. Point cloud saved to: {output_path}")
    else:
        export_colmap(
            predictions=predictions_np,
            image_paths=image_paths,
            image_dir=image_dir,
            output_dir=output_path,
            conf_thres=args.conf_thres,
            mask_black_bg=args.mask_black_bg,
            mask_white_bg=args.mask_white_bg,
            mask_sky=args.mask_sky,
            max_points=args.max_points,
            filter_depth_edges=not args.no_filter_depth_edges,
            depth_edge_rtol=args.depth_edge_rtol,
        )
        logger.info(f"Reconstruction completed. COLMAP model saved to: {output_path}")


if __name__ == "__main__":
    main()
