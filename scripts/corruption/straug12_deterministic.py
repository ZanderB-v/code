#!/usr/bin/env python3
"""Deterministic STRAug-12-inspired corruptions for line OCR evaluation.

The operator set follows the second augmentation round used by Xu et al.:
Curve, Distort, Stretch, Rotate, Perspective, Shrink, TranslateX,
TranslateY, Contrast, Brightness, JpegCompression, and Pixelate.

This is intentionally an evaluation implementation, not random training
augmentation. Every transform is planned from a frozen seed. Geometric
operators expand the canvas when necessary so a corruption cannot remove
labelled characters. The implementation uses Pillow and NumPy only so it
remains portable to the offline OpenOCR server. It is conceptually aligned
with STRAug, but is not claimed to be byte-identical to the OpenCV-TPS
reference.
"""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance


IMPLEMENTATION_ID = "straug12_deterministic_pillow_v2_label_safe"
CORRUPTIONS = (
    "curve",
    "distort",
    "stretch",
    "rotate",
    "perspective",
    "shrink",
    "translate_x",
    "translate_y",
    "contrast",
    "brightness",
    "jpeg_compression",
    "pixelate",
)
SEVERITIES = (1, 2, 3)

_RESAMPLING = getattr(Image, "Resampling", Image)
_TRANSFORM = getattr(Image, "Transform", Image)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_seed(
    global_seed: int,
    sample_id: str,
    corruption: str,
    severity: int,
) -> int:
    payload = f"{global_seed}|{sample_id}|{corruption}|{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _border_color(image: Image.Image) -> tuple[int, int, int]:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    border = np.concatenate(
        (
            array[0, :, :],
            array[-1, :, :],
            array[:, 0, :],
            array[:, -1, :],
        ),
        axis=0,
    )
    color = np.median(border, axis=0).round().astype(np.uint8)
    return tuple(int(value) for value in color)


def _signed(rng: np.random.Generator, magnitude: float) -> float:
    return float(magnitude if rng.random() >= 0.5 else -magnitude)


def plan_corruption(
    corruption: str,
    severity: int,
    size: tuple[int, int],
    seed: int,
) -> dict[str, Any]:
    """Return all sampled parameters needed to reproduce one corruption."""

    if corruption not in CORRUPTIONS:
        raise ValueError(f"Unsupported corruption: {corruption}")
    if severity not in SEVERITIES:
        raise ValueError(f"Severity must be one of {SEVERITIES}: {severity}")

    width, height = size
    rng = np.random.default_rng(seed)
    params: dict[str, Any] = {
        "corruption": corruption,
        "severity": severity,
        "seed": seed,
        "width": width,
        "height": height,
    }

    if corruption == "curve":
        amplitude_frac = (0.04, 0.08, 0.12)[severity - 1]
        params.update(
            {
                "amplitude_px": _signed(
                    rng, max(1.0, amplitude_frac * height)
                ),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
                "waves": float(rng.uniform(0.8, 1.2)),
                "segments": 12,
            }
        )
    elif corruption == "distort":
        x_frac = (0.01, 0.025, 0.04)[severity - 1]
        y_frac = (0.03, 0.06, 0.10)[severity - 1]
        cols, rows = 4, 2
        offsets = []
        for row in range(rows + 1):
            row_offsets = []
            for col in range(cols + 1):
                edge_x = col in (0, cols)
                edge_y = row in (0, rows)
                dx = 0.0 if edge_x else float(rng.uniform(-x_frac, x_frac) * width)
                dy = 0.0 if edge_y else float(rng.uniform(-y_frac, y_frac) * height)
                row_offsets.append([dx, dy])
            offsets.append(row_offsets)
        params.update({"cols": cols, "rows": rows, "offsets": offsets})
    elif corruption == "stretch":
        max_frac = (0.04, 0.08, 0.12)[severity - 1]
        interior = []
        for fraction in (0.25, 0.5, 0.75):
            interior.append(
                float(
                    fraction * width
                    + rng.uniform(-max_frac, max_frac) * width
                )
            )
        interior.sort()
        minimum_gap = max(2.0, 0.08 * width)
        points = [0.0]
        for value in interior:
            points.append(max(points[-1] + minimum_gap, value))
        points.append(float(width))
        for index in range(len(points) - 2, 0, -1):
            points[index] = min(points[index], points[index + 1] - minimum_gap)
        params.update(
            {
                "source_x": [0.0, 0.25 * width, 0.5 * width, 0.75 * width, float(width)],
                "destination_x": points,
            }
        )
    elif corruption == "rotate":
        max_angle = (2.0, 4.0, 7.0)[severity - 1]
        params["angle_deg"] = _signed(
            rng, float(rng.uniform(0.65 * max_angle, max_angle))
        )
        params["canvas_policy"] = "expand_preserve_content"
    elif corruption == "perspective":
        vertical_frac = (0.025, 0.05, 0.085)[severity - 1]
        horizontal_frac = (0.005, 0.012, 0.02)[severity - 1]
        side = "left" if rng.random() < 0.5 else "right"
        vertical = float(rng.uniform(0.7, 1.0) * vertical_frac * height)
        horizontal = float(rng.uniform(0.5, 1.0) * horizontal_frac * width)
        params.update(
            {
                "side": side,
                "vertical_px": vertical,
                "horizontal_px": horizontal,
            }
        )
    elif corruption == "shrink":
        scale = (0.92, 0.84, 0.75)[severity - 1]
        params.update(
            {
                "scale": scale,
                "x_bias": float(rng.uniform(-0.35, 0.35)),
                "y_bias": float(rng.uniform(-0.35, 0.35)),
            }
        )
    elif corruption == "translate_x":
        fraction = (0.015, 0.03, 0.05)[severity - 1]
        params["offset_px"] = _signed(
            rng, float(rng.uniform(0.65, 1.0) * fraction * width)
        )
        params["canvas_policy"] = "expand_preserve_content"
    elif corruption == "translate_y":
        fraction = (0.03, 0.06, 0.10)[severity - 1]
        params["offset_px"] = _signed(
            rng, float(rng.uniform(0.65, 1.0) * fraction * height)
        )
        params["canvas_policy"] = "expand_preserve_content"
    elif corruption == "contrast":
        params["factor"] = (0.75, 0.55, 0.35)[severity - 1]
    elif corruption == "brightness":
        dark, bright = (
            (0.86, 1.14),
            (0.70, 1.30),
            (0.55, 1.45),
        )[severity - 1]
        params["factor"] = float(bright if rng.random() >= 0.5 else dark)
    elif corruption == "jpeg_compression":
        params["quality"] = (75, 50, 30)[severity - 1]
    elif corruption == "pixelate":
        params["scale"] = (0.80, 0.68, 0.55)[severity - 1]

    return params


def _curve_mesh(
    width: int,
    height: int,
    amplitude: float,
    phase: float,
    waves: float,
    segments: int,
) -> list[tuple[tuple[int, int, int, int], tuple[float, ...]]]:
    mesh = []
    for index in range(segments):
        x0 = int(round(index * width / segments))
        x1 = int(round((index + 1) * width / segments))
        y0 = amplitude * math.sin(2.0 * math.pi * waves * x0 / max(width, 1) + phase)
        y1 = amplitude * math.sin(2.0 * math.pi * waves * x1 / max(width, 1) + phase)
        quad = (
            float(x0),
            float(-y0),
            float(x0),
            float(height - y0),
            float(x1),
            float(height - y1),
            float(x1),
            float(-y1),
        )
        mesh.append(((x0, 0, x1, height), quad))
    return mesh


def _distort_mesh(
    width: int,
    height: int,
    cols: int,
    rows: int,
    offsets: list[list[list[float]]],
) -> list[tuple[tuple[int, int, int, int], tuple[float, ...]]]:
    mesh = []
    for row in range(rows):
        for col in range(cols):
            x0 = int(round(col * width / cols))
            x1 = int(round((col + 1) * width / cols))
            y0 = int(round(row * height / rows))
            y1 = int(round((row + 1) * height / rows))
            ul = (x0 + offsets[row][col][0], y0 + offsets[row][col][1])
            ll = (x0 + offsets[row + 1][col][0], y1 + offsets[row + 1][col][1])
            lr = (
                x1 + offsets[row + 1][col + 1][0],
                y1 + offsets[row + 1][col + 1][1],
            )
            ur = (x1 + offsets[row][col + 1][0], y0 + offsets[row][col + 1][1])
            mesh.append(((x0, y0, x1, y1), (*ul, *ll, *lr, *ur)))
    return mesh


def _stretch_mesh(
    width: int,
    height: int,
    source_x: list[float],
    destination_x: list[float],
) -> list[tuple[tuple[int, int, int, int], tuple[float, ...]]]:
    mesh = []
    for index in range(len(source_x) - 1):
        dx0 = int(round(destination_x[index]))
        dx1 = int(round(destination_x[index + 1]))
        sx0 = float(source_x[index])
        sx1 = float(source_x[index + 1])
        if dx1 <= dx0:
            continue
        mesh.append(
            (
                (dx0, 0, dx1, height),
                (sx0, 0.0, sx0, float(height), sx1, float(height), sx1, 0.0),
            )
        )
    return mesh


def _perspective_coefficients(
    destination: list[tuple[float, float]],
    source: list[tuple[float, float]],
) -> tuple[float, ...]:
    matrix = []
    vector = []
    for (x_dst, y_dst), (x_src, y_src) in zip(destination, source):
        matrix.append([x_dst, y_dst, 1, 0, 0, 0, -x_src * x_dst, -x_src * y_dst])
        vector.append(x_src)
        matrix.append([0, 0, 0, x_dst, y_dst, 1, -y_src * x_dst, -y_src * y_dst])
        vector.append(y_src)
    coefficients = np.linalg.solve(
        np.asarray(matrix, dtype=np.float64),
        np.asarray(vector, dtype=np.float64),
    )
    return tuple(float(value) for value in coefficients)


def apply_corruption(image: Image.Image, params: dict[str, Any]) -> Image.Image:
    """Apply one already-planned corruption without any additional randomness."""

    image = image.convert("RGB")
    width, height = image.size
    if (width, height) != (params["width"], params["height"]):
        raise ValueError(
            f"Source size changed: {(width, height)} vs "
            f"{(params['width'], params['height'])}"
        )
    corruption = params["corruption"]
    fill = _border_color(image)

    if corruption == "curve":
        mesh = _curve_mesh(
            width,
            height,
            params["amplitude_px"],
            params["phase"],
            params["waves"],
            params["segments"],
        )
        return image.transform(
            image.size,
            _TRANSFORM.MESH,
            mesh,
            resample=_RESAMPLING.BICUBIC,
            fillcolor=fill,
        )
    if corruption == "distort":
        mesh = _distort_mesh(
            width,
            height,
            params["cols"],
            params["rows"],
            params["offsets"],
        )
        return image.transform(
            image.size,
            _TRANSFORM.MESH,
            mesh,
            resample=_RESAMPLING.BICUBIC,
            fillcolor=fill,
        )
    if corruption == "stretch":
        mesh = _stretch_mesh(
            width,
            height,
            params["source_x"],
            params["destination_x"],
        )
        return image.transform(
            image.size,
            _TRANSFORM.MESH,
            mesh,
            resample=_RESAMPLING.BICUBIC,
            fillcolor=fill,
        )
    if corruption == "rotate":
        return image.rotate(
            params["angle_deg"],
            resample=_RESAMPLING.BICUBIC,
            expand=True,
            fillcolor=fill,
        )
    if corruption == "perspective":
        vertical = params["vertical_px"]
        horizontal = params["horizontal_px"]
        source = [
            (0.0, 0.0),
            (float(width), 0.0),
            (float(width), float(height)),
            (0.0, float(height)),
        ]
        if params["side"] == "left":
            destination = [
                (horizontal, vertical),
                (float(width), 0.0),
                (float(width), float(height)),
                (horizontal, float(height) - vertical),
            ]
        else:
            destination = [
                (0.0, 0.0),
                (float(width) - horizontal, vertical),
                (float(width) - horizontal, float(height) - vertical),
                (0.0, float(height)),
            ]
        coefficients = _perspective_coefficients(destination, source)
        return image.transform(
            image.size,
            _TRANSFORM.PERSPECTIVE,
            coefficients,
            resample=_RESAMPLING.BICUBIC,
            fillcolor=fill,
        )
    if corruption == "shrink":
        scale = params["scale"]
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = image.resize((new_width, new_height), _RESAMPLING.BICUBIC)
        canvas = Image.new("RGB", image.size, fill)
        free_x = width - new_width
        free_y = height - new_height
        x = int(round(free_x * (0.5 + params["x_bias"])))
        y = int(round(free_y * (0.5 + params["y_bias"])))
        x = min(max(0, x), free_x)
        y = min(max(0, y), free_y)
        canvas.paste(resized, (x, y))
        return canvas
    if corruption == "translate_x":
        offset = int(round(params["offset_px"]))
        canvas = Image.new("RGB", (width + abs(offset), height), fill)
        canvas.paste(image, (max(0, offset), 0))
        return canvas
    if corruption == "translate_y":
        offset = int(round(params["offset_px"]))
        canvas = Image.new("RGB", (width, height + abs(offset)), fill)
        canvas.paste(image, (0, max(0, offset)))
        return canvas
    if corruption == "contrast":
        return ImageEnhance.Contrast(image).enhance(params["factor"])
    if corruption == "brightness":
        return ImageEnhance.Brightness(image).enhance(params["factor"])
    if corruption == "jpeg_compression":
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=int(params["quality"]),
            subsampling=2,
            optimize=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    if corruption == "pixelate":
        scale = params["scale"]
        small = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return image.resize(small, _RESAMPLING.BOX).resize(
            image.size,
            _RESAMPLING.NEAREST,
        )
    raise ValueError(f"Unsupported corruption: {corruption}")
