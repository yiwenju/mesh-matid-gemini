"""Camera rig and projection.

Conventions:
  - World frame is the normalized mesh frame: Y-up, bounding-box center at the
    origin, farthest vertex at radius 1.
  - Cameras are OpenCV: x right, y down, z forward. `c2w` maps camera -> world,
    `w2c` is its inverse, `K` is in pixels with the principal point at the
    image center.
  - Azimuth 0 puts the camera on +Z; azimuth increases toward +X.
  - The field of view is the same vertically and horizontally (square images).
"""

from __future__ import annotations

import math

import numpy as np

FOV_DEG = 35.0
# The mesh sits inside the unit sphere, so d * sin(fov / 2) >= 1 keeps it in frame.
DISTANCE = 3.6
ELEVATIONS_DEG = (20.0, 45.0)  # alternate so tops are seen too


def look_at_c2w(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = target - position
    z = z / np.linalg.norm(z)
    up_cam = up - up.dot(z) * z
    up_cam = up_cam / np.linalg.norm(up_cam)
    y = -up_cam  # OpenCV y points down
    x = np.cross(y, z)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, position
    return c2w


def make_cameras(n_views: int = 8, resolution: int = 1024) -> list[dict]:
    focal = (resolution / 2) / math.tan(math.radians(FOV_DEG) / 2)
    K = [[focal, 0, resolution / 2], [0, focal, resolution / 2], [0, 0, 1]]
    views = []
    for i in range(n_views):
        az = 2 * math.pi * i / n_views
        el = math.radians(ELEVATIONS_DEG[i % len(ELEVATIONS_DEG)])
        pos = DISTANCE * np.array(
            [math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)]
        )
        c2w = look_at_c2w(pos, np.zeros(3), np.array([0.0, 1.0, 0.0]))
        views.append(
            {
                "index": i,
                "image_file": f"view_{i:02d}.png",
                "image_rgba": f"view_{i:02d}_rgba.png",
                "width": resolution,
                "height": resolution,
                "azimuth_deg": math.degrees(az),
                "elevation_deg": math.degrees(el),
                "fov_deg": FOV_DEG,
                "K": K,
                "c2w": c2w.tolist(),
                "w2c": np.linalg.inv(c2w).tolist(),
            }
        )
    return views


def project(points: np.ndarray, view: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (u, v) and camera-z depth for world points."""
    w2c = np.array(view["w2c"])
    K = np.array(view["K"])
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3], cam[:, 2]
