"""Tile views into one lossless grid and split a same-shaped image back apart.

Concatenating the views onto one canvas lets the image model see every angle
at once, which keeps colors consistent across views.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image


def grid_shape(n_views: int) -> tuple[int, int]:
    """(rows, cols) that fit every view with no empty cells.

    A square count of 4 or more is a square grid (4 -> 2x2, 9 -> 3x3), another
    even count is two rows (8 -> 2x4), any other count is one row (3 -> 1x3).
    """
    if n_views <= 0:
        raise ValueError("need at least one view")
    side = math.isqrt(n_views)
    if side >= 2 and side * side == n_views:
        return side, side
    if n_views % 2 == 0:
        return 2, n_views // 2
    return 1, n_views


def concat_views(folder: Path, names: list[str]) -> tuple[Path, int, int]:
    """Paste views at native resolution (no resize, crop, or gap) into views_RxC.png."""
    rows, cols = grid_shape(len(names))
    images = [Image.open(folder / n).convert("RGB") for n in names]
    width, height = images[0].size
    if any(im.size != (width, height) for im in images):
        raise ValueError("all views must have the same size")
    canvas = Image.new("RGB", (cols * width, rows * height))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        canvas.paste(im, (c * width, r * height))
    out = folder / f"views_{rows}x{cols}.png"
    canvas.save(out, format="PNG", compress_level=3)
    # Confirm the paste lost no pixels.
    grid = np.asarray(Image.open(out))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        tile = grid[r * height : (r + 1) * height, c * width : (c + 1) * width]
        if not np.array_equal(tile, np.asarray(im)):
            raise AssertionError(f"{names[i]} does not match its tile in {out.name}")
    return out, rows, cols


def split_grid(
    grid: Image.Image,
    folder: Path,
    names: list[str],
    rows: int,
    cols: int,
    tile_size: tuple[int, int],
    suffix: str = "_matcolor",
) -> list[Path]:
    """Cut `grid` into one tile per view, resized to `tile_size`, as <view><suffix>.png."""
    # Cell edges are rounded to whole pixels when the canvas does not divide evenly.
    xs = [round(c * grid.size[0] / cols) for c in range(cols + 1)]
    ys = [round(r * grid.size[1] / rows) for r in range(rows + 1)]
    written = []
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        tile = grid.crop((xs[c], ys[r], xs[c + 1], ys[r + 1]))
        if tile.size != tile_size:
            tile = tile.resize(tile_size, Image.Resampling.LANCZOS)
        out = folder / f"{Path(name).stem}{suffix}.png"
        tile.save(out, format="PNG", compress_level=3)
        written.append(out)
    return written
