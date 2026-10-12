"""python -m matid_gemini MESH OUT_DIR

Mesh -> clay renders -> one concatenated grid -> Gemini material-ID colors ->
one color map per view. Needs Blender (BLENDER_BIN or `blender` on PATH) and,
unless --no-gemini, GEMINI_API_KEY or GOOGLE_API_KEY.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from . import gemini, grid, render


def run(
    mesh: Path,
    out_dir: Path,
    n_views: int = 8,
    resolution: int = 1024,
    samples: int = 16,
    shading: str = "clay",
    use_gemini: bool = True,
    model: str = gemini.MODEL,
    image_size: str = gemini.IMAGE_SIZE,
) -> list[Path]:
    views = render.render_views(mesh, out_dir, n_views, resolution, samples, shading)
    names = [v["image_file"] for v in views]
    grid_path, rows, cols = grid.concat_views(out_dir, names)
    print(f"rendered {len(names)} views, wrote {grid_path}")
    if not use_gemini:
        return []
    print(f"gemini: {model}, {image_size}")
    painted = gemini.paint_material_colors(grid_path, model, image_size)
    painted.save(out_dir / f"views_{rows}x{cols}_matcolor.png", format="PNG", compress_level=3)
    outputs = grid.split_grid(painted, out_dir, names, rows, cols, (resolution, resolution))
    print(f"wrote {len(outputs)} material-color maps in {out_dir}")
    return outputs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mesh", type=Path, help="obj, glb, gltf, ply, stl, fbx, or blend")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--views", type=int, default=8, help="number of views (default 8 -> 2x4 grid)")
    p.add_argument("--resolution", type=int, default=1024, help="pixels per view (default 1024)")
    p.add_argument("--samples", type=int, default=16, help="Cycles samples (default 16)")
    p.add_argument("--shading", choices=["clay", "textured"], default="clay")
    p.add_argument("--no-gemini", action="store_true", help="stop after rendering and concatenating")
    p.add_argument("--model", default=gemini.MODEL)
    p.add_argument("--image-size", default=gemini.IMAGE_SIZE, help="512, 1K, 2K, or 4K")
    a = p.parse_args()
    run(a.mesh, a.out_dir, a.views, a.resolution, a.samples, a.shading,
        not a.no_gemini, a.model, a.image_size)


if __name__ == "__main__":
    main()
