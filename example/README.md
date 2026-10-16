# Example: sneakers (two shoes, 39.5k triangles)

One full run of the pipeline: GLB mesh -> 8 clay renders -> 2x4 grid -> Gemini material-ID colors -> 8 per-view maps.

```bash
python example/run_example.py shoes.glb out/shoes                    # run + check + sheets
python example/run_example.py shoes.glb out/shoes --skip-pipeline    # re-check an existing run
```

`run_example.py` calls `matid_gemini.run` with the default settings (8 views, 1024 px, 16 Cycles samples, `gemini-3.1-flash-image` at 4K), then checks the outputs and builds the comparison sheets. The input mesh is not included in this repo; it comes from a dataset asset, so supply your own mesh to reproduce the run. Gemini's output is not deterministic, so a new run will pick different colors.

## Results (`results/`)

| file | content |
|---|---|
| `view_00.png` ... `view_07.png` | clay renders sent to Gemini |
| `views_2x4.png` | their lossless 2x4 concatenation (4096x2048) |
| `views_2x4_matcolor_half.png` | Gemini's canvas at half size (the full 5760x2880 canvas is 10 MB; the per-view maps below are its full-resolution split) |
| `view_00_matcolor.png` ... `view_07_matcolor.png` | per-view material-color maps, 1024x1024 |
| `sheet_views_0-3.png`, `sheet_views_4-7.png` | per view: render, Gemini map, silhouette difference (gray = both, red = painted outside the silhouette, blue = silhouette left unpainted) |
| `cameras.json`, `views_metadata.json` | intrinsics and extrinsics per view, silhouette check |
| `report.txt` | the numeric checks below |

## What the checks show

- Camera round-trip: 100% of mesh vertices project inside the rendered silhouette in all 8 views.
- The grid sent to Gemini is pixel-identical to the 8 renders; Gemini's canvas has the same 2:1 aspect ratio; all split tiles are 1024x1024.
- Gemini used 5 colors: cyan (upper), red (sole), yellow (laces, orange eyelets), magenta (tongue and inner lining), dark green (insole). Each color stays on the same part in all 8 views, including the back and interior views.
- Painted foreground vs the render's silhouette: IoU 0.977-0.986 in 7 views and 0.921 in view 3, where Gemini's panel is shifted by a few pixels. Most views also show a 1-2 px halo along some edges, and the background has faint gray marks near panel borders. Treat the maps as approximate: quantize, clean, and align them to the silhouette before using them as labels.
