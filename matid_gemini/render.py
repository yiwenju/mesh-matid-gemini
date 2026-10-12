"""Render views of a mesh with Blender and verify the cameras."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .cameras import make_cameras, project

BLENDER_SCRIPT = Path(__file__).with_name("blender_render.py")
BACKGROUND = (255, 255, 255)


def find_blender() -> str:
    """$BLENDER_BIN, else `blender` on PATH."""
    path = os.environ.get("BLENDER_BIN") or shutil.which("blender")
    if not path:
        raise FileNotFoundError("Blender not found: set BLENDER_BIN or put `blender` on PATH")
    return path


def silhouette_check(out_dir: Path, views: list[dict], tol_px: int = 3) -> dict:
    """Camera round-trip: projected vertices must land inside the rendered alpha.

    Catches axis, handedness, and fov mix-ups, which otherwise fail silently.
    """
    verts = np.load(out_dir / "mesh.npz")["verts"].astype(np.float64)
    report = {}
    for view in views:
        alpha = Image.open(out_dir / view["image_rgba"]).getchannel("A")
        mask = np.asarray(alpha.filter(ImageFilter.MaxFilter(2 * tol_px + 1))) > 8
        uv, depth = project(verts, view)
        h, w = mask.shape
        ij = np.floor(uv).astype(int)
        ok = (depth > 0) & (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
        inside = np.zeros(len(verts), dtype=bool)
        inside[ok] = mask[ij[ok, 1], ij[ok, 0]]
        report[view["index"]] = {"frac_inside": float(inside.mean())}
    return report


def composite_rgb(out_dir: Path, views: list[dict]) -> None:
    for view in views:
        rgba = Image.open(out_dir / view["image_rgba"]).convert("RGBA")
        bg = Image.new("RGBA", rgba.size, BACKGROUND + (255,))
        Image.alpha_composite(bg, rgba).convert("RGB").save(out_dir / view["image_file"])


def render_views(
    mesh: Path,
    out_dir: Path,
    n_views: int = 8,
    resolution: int = 1024,
    samples: int = 16,
    shading: str = "clay",
) -> list[dict]:
    """Render `n_views` clay views; writes view_XX.png, cameras.json, mesh.npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    views = make_cameras(n_views, resolution)
    cameras_path = out_dir / "cameras.json"
    cameras_path.write_text(json.dumps({"views": views}, indent=1))
    args = ["--out", str(out_dir), "--cameras", str(cameras_path),
            "--samples", str(samples), "--shading", shading]
    if mesh.suffix.lower() == ".blend":
        cmd = [find_blender(), "-b", str(mesh), "--python", str(BLENDER_SCRIPT), "--", *args]
    else:
        cmd = [find_blender(), "-b", "--factory-startup", "--python", str(BLENDER_SCRIPT),
               "--", "--mesh", str(mesh.resolve()), *args]
    with open(out_dir / "blender.log", "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    composite_rgb(out_dir, views)
    report = silhouette_check(out_dir, views)
    worst = min(r["frac_inside"] for r in report.values())
    if worst < 0.98:
        raise RuntimeError(f"camera round-trip failed (worst view {worst:.3f} inside); see {out_dir}")
    (out_dir / "views_metadata.json").write_text(
        json.dumps({"source": str(mesh), "shading": shading, "views": views,
                    "silhouette_check": report}, indent=1)
    )
    return views
