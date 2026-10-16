# mesh-matid-gemini

Given a 3D mesh, get a **material-ID color map** for every view of it:

1. **Render** the mesh from 8 fixed cameras as untextured gray ("clay") images with Blender.
2. **Concatenate** the views into one lossless grid (8 views -> 2x4). One canvas lets the image model see every angle at once, which keeps each material's color consistent across views.
3. **Generate** the material map with Gemini (Nano Banana 2): one edit that repaints each distinct material in its own flat color.
4. **Split** the result back into one image per view, matching the render resolution.

```
mesh.glb -> view_00.png ... view_07.png -> views_2x4.png
         -> (Gemini) views_2x4_matcolor.png -> view_00_matcolor.png ... view_07_matcolor.png
```

The color map carries no material *identity* (no color -> name), only "same color = same material". Colors are approximate and not pixel-aligned to the mesh; quantize and clean them before using them as labels.

## Setup

```bash
pip install -r requirements.txt          # numpy, pillow, google-genai
export BLENDER_BIN=/path/to/blender      # or have `blender` on PATH (4.2+; tested on 4.5)
export GEMINI_API_KEY=...                # or GOOGLE_API_KEY; never commit it
```

Blender only does the rendering, so any recent build works. Rendering uses Cycles and picks a GPU (OptiX or CUDA) if Blender can see one; otherwise it falls back to CPU. The first GPU run compiles kernels and can take several minutes; later runs are fast.

## Usage

```bash
python -m matid_gemini model.glb out/model
python -m matid_gemini model.obj out/model --views 8 --resolution 1024
python -m matid_gemini model.glb out/model --no-gemini     # render + concat only
```

Inputs: `.obj`, `.glb`, `.gltf`, `.ply`, `.stl`, `.fbx`, or `.blend` (all mesh objects in the file are merged). Y-up files keep their orientation. `--shading textured` renders the file's own materials instead of clay.

Options: `--views N` (3 -> 1x3, 4 -> 2x2, 6 -> 2x3, 8 -> 2x4, 9 -> 3x3), `--resolution` (pixels per view, default 1024), `--samples` (Cycles, default 16), `--model`, `--image-size`.

From Python:

```python
from pathlib import Path
from matid_gemini.__main__ import run
run(Path("model.glb"), Path("out/model"))
```

## Output folder

| file | content |
|---|---|
| `view_XX.png` | clay render on white, 1024 px |
| `view_XX_rgba.png` | same render with transparent background (silhouette mask) |
| `views_RxC.png` | lossless concatenation sent to Gemini |
| `views_RxC_matcolor.png` | Gemini output, full canvas |
| `view_XX_matcolor.png` | material-color map per view, same size as the render |
| `cameras.json`, `views_metadata.json` | per-view intrinsics `K`, `c2w`, `w2c`, plus the silhouette check |
| `mesh.npz` | normalized vertices and triangles, and per-face material index/names if the input had materials |

## Cameras

World frame is the mesh normalized to Y-up, bounding-box center at the origin, farthest vertex at radius 1. Cameras are OpenCV (x right, y down, z forward); `K` is in pixels with the principal point at the image center. The 8 cameras sit at azimuths every 45 degrees (azimuth 0 on +Z, increasing toward +X) with elevations alternating between 20 and 45 degrees, 3.6 units from the origin, 35 degree field of view.

Every run checks the cameras: projected mesh vertices must land inside the rendered silhouette (>= 98% in every view), otherwise it stops. This catches axis, handedness, and field-of-view mix-ups, which otherwise fail silently.

## Example

`example/` has a complete run on a sneaker mesh: the script (`run_example.py`), all renders and Gemini maps, comparison sheets, and the numeric checks. See `example/README.md`.

## Notes and limits

- Gemini sees only gray shapes, so it separates materials that differ in geometry (a shoe's sole, laces, and upper) and merges ones that differ only in appearance (two woods, a printed label). It also tends to under-segment. Expect to inspect results.
- Output is not deterministic; the same mesh can give different colors on different runs.
- The result is checked to have the input's aspect ratio and is rejected rather than stretched if it doesn't.
- The prompt and layout lock are in `matid_gemini/gemini.py`.
