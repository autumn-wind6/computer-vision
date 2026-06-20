#!/usr/bin/env python3
"""Render an orbit walkthrough video from an ASCII XYZRGB point-cloud PLY.

Blender/Cycles cannot see bare vertices; this script projects colored points
directly and works on the server or on a Mac with only numpy + opencv-python.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cvfinal.ply import read_ascii_xyzrgb_ply


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = read_ascii_xyzrgb_ply(path)
    points = np.array([[v.x, v.y, v.z] for v in vertices], dtype=np.float64)
    colors = np.array([[v.red, v.green, v.blue] for v in vertices], dtype=np.uint8)
    return points, colors


def scene_center_radius(points: np.ndarray) -> tuple[np.ndarray, float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float(np.linalg.norm(maxs - mins) * 0.5), 1.0)
    return center, radius


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    up = np.array([0.0, 1.0, 0.0]) if up is None else up.astype(np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8
    true_up = np.cross(right, forward)
    # Camera +Z points toward the target so in-front points have positive depth.
    rotation = np.stack([right, true_up, forward], axis=0)
    translation = -rotation @ eye
    return rotation, translation


def render_frame(
    points: np.ndarray,
    colors: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    width: int,
    height: int,
    focal: float,
    point_radius: int,
    bg_color: tuple[int, int, int],
) -> np.ndarray:
    cam = (rotation @ points.T).T + translation
    depth = cam[:, 2]
    visible = depth > 0.05
    if not np.any(visible):
        return np.full((height, width, 3), bg_color, dtype=np.uint8)

    cam = cam[visible]
    depth = depth[visible]
    rgb = colors[visible]

    cx, cy = width * 0.5, height * 0.5
    xs = (focal * cam[:, 0] / cam[:, 2] + cx).astype(np.int32)
    ys = (focal * cam[:, 1] / cam[:, 2] + cy).astype(np.int32)

    in_view = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys, depth, rgb = xs[in_view], ys[in_view], depth[in_view], rgb[in_view]
    order = np.argsort(depth)

    image = np.full((height, width, 3), bg_color, dtype=np.uint8)
    if point_radius <= 1:
        image[ys[order], xs[order]] = rgb[order]
        return image

    for x, y, color in zip(xs[order], ys[order], rgb[order]):
        cv2.circle(image, (int(x), int(y)), point_radius, color.tolist(), -1, lineType=cv2.LINE_AA)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a fused point-cloud PLY to an MP4 orbit video.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--resolution-x", type=int, default=1280)
    parser.add_argument("--resolution-y", type=int, default=720)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--subsample", type=int, default=0, help="Keep every Nth point (0 = keep all).")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    print(f"Loading {args.input} ...")
    points, colors = load_points(args.input)
    if args.subsample > 1:
        points = points[:: args.subsample]
        colors = colors[:: args.subsample]
    print(f"  {len(points):,} points")

    center, radius = scene_center_radius(points)
    width, height = args.resolution_x, args.resolution_y
    focal = max(width, height) * 0.9
    frame_count = max(2, int(round(args.fps * args.duration)))
    distance = radius * 2.8
    height_offset = radius * 0.9
    bg = (8, 8, 10)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {args.output}")

    for frame_idx in range(frame_count):
        theta = 2.0 * math.pi * frame_idx / max(frame_count - 1, 1)
        eye = center + np.array(
            [math.cos(theta) * distance, height_offset, math.sin(theta) * distance],
            dtype=np.float64,
        )
        rotation, translation = look_at(eye, center)
        image = render_frame(
            points, colors, rotation, translation, width, height, focal, args.point_radius, bg
        )
        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if frame_idx % max(args.fps, 1) == 0:
            print(f"  frame {frame_idx + 1}/{frame_count}")

    writer.release()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
