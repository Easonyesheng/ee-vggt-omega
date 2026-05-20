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

from visual_util import predictions_to_glb
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
) -> "trimesh.PointCloud":
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

    sky_target_dir = None
    temp_dir = None
    try:
        if mask_sky:
            if os.path.basename(image_dir) == "images" and os.path.isdir(os.path.dirname(image_dir)):
                sky_target_dir = os.path.dirname(image_dir)
            else:
                temp_dir = tempfile.mkdtemp(prefix="vggt_omega_")
                os.symlink(os.path.abspath(image_dir), os.path.join(temp_dir, "images"))
                sky_target_dir = temp_dir

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
    return point_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT-Omega reconstruction and export a point-cloud PLY.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to VGGT-Omega checkpoint (.pt).")
    parser.add_argument("--input-dir", required=True, help="Directory containing input images.")
    parser.add_argument("--output", default=None, help="Output PLY path (default: <input-dir>/scene.ply).")
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


    output_path = args.output or os.path.join(args.input_dir, "scene.ply")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    point_cloud = build_point_cloud(
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
    point_cloud.export(output_path)
    logger.info(f"Reconstruction completed. Point cloud saved to: {output_path}")


if __name__ == "__main__":
    main()
