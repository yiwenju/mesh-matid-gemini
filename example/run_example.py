"""Run the pipeline on one mesh, then check the results and build comparison sheets.

    python example/run_example.py model.glb out/model
    python example/run_example.py model.glb out/model --skip-pipeline   # re-check an existing run

Checks written to <out_dir>/report.txt:
  - camera round-trip (from the pipeline's silhouette check)
  - the grid sent to Gemini is pixel-identical to the 8 renders
  - Gemini's canvas has the input's aspect ratio and the split tiles match the renders
  - per view: IoU between Gemini's painted foreground and the render's silhouette,
    and the dominant colors
Sheets <out_dir>/sheet_views_*.png show, for 4 views each: the clay render, Gemini's
map, and a silhouette difference (gray = both, red = painted outside the silhouette,
blue = silhouette left unpainted).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matid_gemini.__main__ import run  # noqa: E402
from matid_gemini.grid import grid_shape  # noqa: E402

BG_THRESHOLD = 235  # a pixel is "painted" if any channel is below this


def dominant_colors(rgb: np.ndarray, mask: np.ndarray, top: int = 5) -> list[tuple[tuple[int, int, int], float]]:
    px = rgb[mask].astype(int) // 32  # coarse 8-level bins per channel
    key = px[:, 0] * 64 + px[:, 1] * 8 + px[:, 2]
    uniq, counts = np.unique(key, return_counts=True)
    out = []
    for j in np.argsort(-counts)[:top]:
        if counts[j] / counts.sum() > 0.01:
            color = tuple(int(v) * 32 + 16 for v in (uniq[j] // 64, (uniq[j] // 8) % 8, uniq[j] % 8))
            out.append((color, round(float(100 * counts[j] / counts.sum()), 1)))
    return out


def check(out_dir: Path) -> str:
    meta = json.loads((out_dir / "views_metadata.json").read_text())
    n = len(meta["views"])
    lines = [f"views: {n}"]
    sil = {k: round(v["frac_inside"], 3) for k, v in meta["silhouette_check"].items()}
    lines.append(f"camera round-trip, fraction of vertices inside silhouette: {sil}")

    rows, cols = grid_shape(n)
    grid = np.asarray(Image.open(out_dir / f"views_{rows}x{cols}.png"))
    h, w = grid.shape[0] // rows, grid.shape[1] // cols
    same = all(
        np.array_equal(
            grid[(i // cols) * h : (i // cols + 1) * h, (i % cols) * w : (i % cols + 1) * w],
            np.asarray(Image.open(out_dir / f"view_{i:02d}.png").convert("RGB")),
        )
        for i in range(n)
    )
    lines.append(f"concatenated grid pixel-identical to the {n} renders: {same}")

    canvas = Image.open(out_dir / f"views_{rows}x{cols}_matcolor.png")
    lines.append(
        f"gemini canvas {canvas.size[0]}x{canvas.size[1]} (aspect {canvas.size[0] / canvas.size[1]:.3f}, "
        f"input {grid.shape[1] / grid.shape[0]:.3f})"
    )
    sizes = {Image.open(out_dir / f"view_{i:02d}_matcolor.png").size for i in range(n)}
    lines.append(f"split tile sizes: {sorted(sizes)} (render {w}x{h})")

    lines.append("view  IoU(painted fg, render silhouette)  dominant colors (rgb, % of silhouette interior)")
    for i in range(n):
        rgb = np.asarray(Image.open(out_dir / f"view_{i:02d}_matcolor.png").convert("RGB"))
        alpha = np.asarray(Image.open(out_dir / f"view_{i:02d}_rgba.png").getchannel("A")) > 127
        fg = rgb.min(2) < BG_THRESHOLD
        iou = (alpha & fg).sum() / (alpha | fg).sum()
        interior = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(9))) > 0
        lines.append(f"  {i}   {iou:.3f}   {dominant_colors(rgb, interior)}")
    return "\n".join(lines)


def sheets(out_dir: Path, n_views: int, tile: int = 512, per_sheet: int = 4) -> list[Path]:
    written = []
    for start in range(0, n_views, per_sheet):
        idx = list(range(start, min(start + per_sheet, n_views)))
        sheet = Image.new("RGB", (tile * len(idx), tile * 3), "white")
        for c, i in enumerate(idx):
            clay = Image.open(out_dir / f"view_{i:02d}.png").convert("RGB")
            painted = Image.open(out_dir / f"view_{i:02d}_matcolor.png").convert("RGB")
            alpha = np.asarray(Image.open(out_dir / f"view_{i:02d}_rgba.png").getchannel("A")) > 127
            fg = np.asarray(painted).min(2) < BG_THRESHOLD
            diff = np.full((*alpha.shape, 3), 255, np.uint8)
            diff[alpha & fg] = (200, 200, 200)
            diff[fg & ~alpha] = (255, 0, 0)
            diff[alpha & ~fg] = (0, 0, 255)
            for r, im in enumerate((clay, painted, Image.fromarray(diff))):
                sheet.paste(im.resize((tile, tile), Image.Resampling.LANCZOS), (c * tile, r * tile))
            ImageDraw.Draw(sheet).text((c * tile + 8, 6), f"view {i}", fill=(0, 0, 0))
        path = out_dir / f"sheet_views_{idx[0]}-{idx[-1]}.png"
        sheet.save(path)
        written.append(path)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mesh", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--skip-pipeline", action="store_true", help="only check/sheet an existing run")
    a = p.parse_args()
    if not a.skip_pipeline:
        run(a.mesh, a.out_dir)
    report = check(a.out_dir)
    (a.out_dir / "report.txt").write_text(report + "\n")
    print(report)
    n = len(json.loads((a.out_dir / "views_metadata.json").read_text())["views"])
    for path in sheets(a.out_dir, n):
        print("wrote", path)


if __name__ == "__main__":
    main()
