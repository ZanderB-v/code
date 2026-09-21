#!/usr/bin/env python3
"""Postprocess raw synthetic line images into final training images."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def default_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    root = script_path.parents[2]
    return {"dataset_dir": root / "03_synthetic_generation" / "synthetic_smoke_v1"}


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=defaults["dataset_dir"])
    parser.add_argument("--metadata-raw", type=Path, default=None)
    parser.add_argument("--target-height", type=int, default=48)
    parser.add_argument("--max-aspect", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--allow-hard", action="store_true")
    parser.add_argument("--difficulty-profile", choices=("safe", "formal_diverse"), default="safe")
    parser.add_argument("--overwrite-final", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def choose_difficulty(
    rng: random.Random,
    language: str = "",
    allow_hard: bool = False,
    profile: str = "safe",
) -> str:
    roll = rng.random()
    if profile == "formal_diverse":
        if language == "ug":
            if roll < 0.50:
                return "clear"
            if not allow_hard or roll < 0.93:
                return "medium"
            return "hard"
        if roll < 0.45:
            return "clear"
        if not allow_hard or roll < 0.90:
            return "medium"
        return "hard"
    if language == "ug":
        if roll < 0.65:
            return "clear"
        return "medium"
    if not allow_hard:
        if roll < 0.60:
            return "clear"
        return "medium"
    if roll < 0.60:
        return "clear"
    if roll < 0.95:
        return "medium"
    return "hard"


def resize_height_keep_aspect(img: Image.Image, target_height: int) -> Image.Image:
    w, h = img.size
    if h <= 0:
        return img
    new_w = max(1, round(w * target_height / h))
    return img.resize((new_w, target_height), Image.Resampling.LANCZOS)


def add_noise(img: Image.Image, rng: random.Random, sigma: float) -> Image.Image:
    try:
        import numpy as np

        arr = np.asarray(img.convert("RGB")).astype("float32")
        np_rng = np.random.default_rng(rng.randrange(0, 2**63))
        noise = np_rng.normal(0.0, sigma, size=arr.shape[:2]).astype("float32")
        arr += noise[:, :, None]
        arr = np.clip(arr, 0, 255).astype("uint8")
        return Image.fromarray(arr, "RGB")
    except Exception:
        return img


def motion_blur(img: Image.Image, radius: int, horizontal: bool = True) -> Image.Image:
    if radius <= 1:
        return img
    size = radius * 2 + 1
    weights = [0.0] * (size * size)
    if horizontal:
        mid = radius
        for x in range(size):
            weights[mid * size + x] = 1.0 / size
    else:
        mid = radius
        for y in range(size):
            weights[y * size + mid] = 1.0 / size
    return img.filter(ImageFilter.Kernel((size, size), weights, scale=1.0))


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    from io import BytesIO

    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_degradation(img: Image.Image, difficulty: str, rng: random.Random) -> tuple[Image.Image, dict]:
    img = img.convert("RGB")
    meta = {
        "difficulty": difficulty,
        "jpeg_quality": 95,
        "gaussian_blur": 0.0,
        "motion_blur": 0,
        "downsample_scale": 1.0,
        "brightness": 1.0,
        "contrast": 1.0,
        "color_saturation": 1.0,
        "noise_sigma": 0.0,
        "rotation_post": 0.0,
    }
    if difficulty == "clear":
        meta["jpeg_quality"] = rng.randint(88, 96)
        if rng.random() < 0.20:
            meta["brightness"] = rng.uniform(0.94, 1.06)
        if rng.random() < 0.20:
            meta["contrast"] = rng.uniform(0.94, 1.08)
    elif difficulty == "medium":
        meta["jpeg_quality"] = rng.randint(76, 92)
        meta["brightness"] = rng.uniform(0.88, 1.12)
        meta["contrast"] = rng.uniform(0.86, 1.16)
        if rng.random() < 0.35:
            meta["color_saturation"] = rng.uniform(0.88, 1.14)
        if rng.random() < 0.45:
            meta["gaussian_blur"] = rng.uniform(0.20, 0.60)
        if rng.random() < 0.30:
            meta["noise_sigma"] = rng.uniform(1.5, 4.5)
        if rng.random() < 0.25:
            meta["downsample_scale"] = rng.uniform(0.68, 0.88)
    else:
        meta["jpeg_quality"] = rng.randint(64, 82)
        meta["brightness"] = rng.uniform(0.82, 1.18)
        meta["contrast"] = rng.uniform(0.82, 1.24)
        meta["color_saturation"] = rng.uniform(0.76, 1.24)
        if rng.random() < 0.70:
            meta["gaussian_blur"] = rng.uniform(0.30, 0.80)
        if rng.random() < 0.35:
            meta["motion_blur"] = rng.choice([1, 2])
        if rng.random() < 0.45:
            meta["noise_sigma"] = rng.uniform(3.0, 7.0)
        if rng.random() < 0.55:
            meta["downsample_scale"] = rng.uniform(0.62, 0.82)
        if rng.random() < 0.22:
            meta["rotation_post"] = rng.uniform(-1.2, 1.2)

    if meta["downsample_scale"] < 0.99:
        w, h = img.size
        small = (max(1, round(w * meta["downsample_scale"])), max(1, round(h * meta["downsample_scale"])))
        img = img.resize(small, Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BICUBIC)
    if meta["gaussian_blur"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=meta["gaussian_blur"]))
    if meta["motion_blur"] > 0:
        img = motion_blur(img, meta["motion_blur"], horizontal=rng.random() < 0.75)
    if meta["brightness"] != 1.0:
        img = ImageEnhance.Brightness(img).enhance(meta["brightness"])
    if meta["contrast"] != 1.0:
        img = ImageEnhance.Contrast(img).enhance(meta["contrast"])
    if meta["color_saturation"] != 1.0:
        img = ImageEnhance.Color(img).enhance(meta["color_saturation"])
    if meta["noise_sigma"] > 0:
        img = add_noise(img, rng, meta["noise_sigma"])
    if abs(meta["rotation_post"]) > 0:
        img = img.rotate(meta["rotation_post"], resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(245, 245, 245))
        img = ImageOps.crop(img, border=max(0, min(3, min(img.size) // 30)))
    img = jpeg_roundtrip(img, meta["jpeg_quality"])
    return img, meta


def write_labels(dataset_dir: Path, rows: list[dict]) -> None:
    labels_dir = dataset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    all_lines = []
    for lang in ("zh", "ug", "kk"):
        lines = [f"{row['image']}\t{row['ctc_text']}" for row in rows if row["language"] == lang]
        (labels_dir / f"train_{lang}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        all_lines.extend(lines)
    (labels_dir / "train_all.txt").write_text("\n".join(all_lines) + ("\n" if all_lines else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metadata_raw = args.metadata_raw or (args.dataset_dir / "metadata_raw.jsonl")
    raw_rows = read_jsonl(metadata_raw)
    final_root = args.dataset_dir / "synthetic_final"
    if final_root.exists() and args.overwrite_final:
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows = []
    failures = []
    for row in raw_rows:
        raw_path = args.dataset_dir / row["raw_image"]
        try:
            img = Image.open(raw_path).convert("RGB")
        except Exception as exc:
            failures.append({"id": row.get("id"), "reason": f"raw_read_error:{exc}"})
            continue
        difficulty = choose_difficulty(rng, row.get("language", ""), args.allow_hard, args.difficulty_profile)
        img, degradation = apply_degradation(img, difficulty, rng)
        img = resize_height_keep_aspect(img, args.target_height)
        aspect = img.width / max(1, img.height)
        if aspect > args.max_aspect:
            failures.append({"id": row.get("id"), "reason": f"aspect_gt_{args.max_aspect}", "width": img.width, "height": img.height})
            continue
        rel = Path("synthetic_final") / row["language"] / f"{row['id']}.jpg"
        out_path = args.dataset_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, quality=degradation["jpeg_quality"], optimize=True)
        meta = dict(row)
        meta.update(degradation)
        meta["difficulty_profile"] = args.difficulty_profile
        meta["image"] = rel.as_posix()
        meta["final_width"] = img.width
        meta["final_height"] = img.height
        meta["postprocessor"] = "07_postprocess_images.py"
        rows.append(meta)

    (args.dataset_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    (args.dataset_dir / "postprocess_failures.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in failures) + ("\n" if failures else ""),
        encoding="utf-8",
    )
    write_labels(args.dataset_dir, rows)
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "raw_rows": len(raw_rows),
        "final_rows": len(rows),
        "failures": len(failures),
        "per_lang": dict(Counter(row["language"] for row in rows)),
        "difficulty_profile": args.difficulty_profile,
        "difficulty_counts": dict(Counter(row["difficulty"] for row in rows)),
        "target_height": args.target_height,
        "outputs": {
            "metadata": str(args.dataset_dir / "metadata.jsonl"),
            "labels": str(args.dataset_dir / "labels"),
            "synthetic_final": str(final_root),
        },
    }
    (args.dataset_dir / "summary_postprocess.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
