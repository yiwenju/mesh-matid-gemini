"""Paint a flat color per material on the concatenated views with Gemini.

Requires GEMINI_API_KEY or GOOGLE_API_KEY.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types

MODEL = "gemini-3.1-flash-image"  # Nano Banana 2
IMAGE_SIZE = "4K"

# Relative aspect-ratio error allowed between the model output and the input.
ASPECT_TOL = 0.01

PROMPT_MATERIAL_COLORS = (
    "You are a professional game artist who can model great 3D models. "
    "These are untextured gray renders of a 3D model. Repaint them as a "
    "material map: judge from the shapes and the object's function which parts "
    "would be made of different materials, and give every part made of a "
    "different material its own flat, distinct, highly saturated color "
    "(for example red, green, blue, yellow, magenta, cyan, orange), and every "
    "part made of the same material gets exactly the same color. Fill each "
    "region with one uniform color: no shading, lighting, shadows, textures, "
    "gradients, highlights, outlines, or text. Keep the geometry, silhouette, "
    "and background exactly as in the original; only change the surface colors. "
    "Use as few colors as the real material count requires, and do not split "
    "one material into several colors."
)

# Splitting the result back into views assumes every panel stays in place.
LAYOUT_LOCK = (
    "This image is one canvas of separate view panels showing the same object "
    "from different angles. Keep the same canvas aspect ratio, the same panel "
    "arrangement, and the same framing inside every panel. Do not crop, "
    "rearrange, or move content between panels. A material must have the same "
    "color in every panel."
)


def _image_from_response(response) -> Image.Image:
    texts: list[str] = []
    parts = getattr(response, "parts", None) or []
    if not parts:
        candidates = getattr(response, "candidates", None) or []
        if candidates and candidates[0].content is not None:
            parts = candidates[0].content.parts or []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
        generated = part.as_image() if hasattr(part, "as_image") else None
        if generated is not None and generated.image_bytes:
            return Image.open(io.BytesIO(generated.image_bytes)).convert("RGB")
    detail = "\n".join(texts).strip() or "no text"
    raise RuntimeError(f"model returned no image ({detail})")


def paint_material_colors(
    grid_path: Path,
    model: str = MODEL,
    image_size: str = IMAGE_SIZE,
    prompt: str = PROMPT_MATERIAL_COLORS,
) -> Image.Image:
    """One Gemini edit of the grid image; returns the painted canvas.

    1:1 is sent as the aspect ratio for square grids. The API rejects ratios
    like "2:1", so none is sent otherwise and the model matches the input's
    shape; a shape mismatch in the result is rejected rather than stretched.
    """
    with Image.open(grid_path) as source:
        source_rgb = source.convert("RGB")
    aspect = "1:1" if source_rgb.size[0] == source_rgb.size[1] else None
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=[f"{prompt}\n\n{LAYOUT_LOCK}", source_rgb],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(image_size=image_size, aspect_ratio=aspect),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    painted = _image_from_response(response)
    got = painted.size[0] / painted.size[1]
    want = source_rgb.size[0] / source_rgb.size[1]
    if abs(got - want) / want > ASPECT_TOL:
        raise ValueError(
            f"model returned {painted.size[0]}x{painted.size[1]} for a "
            f"{source_rgb.size[0]}x{source_rgb.size[1]} canvas; refusing to stretch it"
        )
    return painted
