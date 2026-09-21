#!/usr/bin/env python3
"""Build a train-only background library for synthetic meme line rendering."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageStat


def default_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    root = script_path.parents[2]
    workspace = root.parents[2]
    return {
        "workspace_root": workspace,
        "index_json": workspace / "final_multilingual_meme_ocr_dataset" / "train_ug.json",
        "output_dir": root / "03_synthetic_generation" / "background_library_v1",
    }


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=defaults["workspace_root"])
    parser.add_argument("--index-json", type=Path, default=defaults["index_json"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--bbox-field", default="render_layout_bbox")
    parser.add_argument("--fallback-bbox-fields", default="render_bbox,bbox")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--pad-x-min", type=int, default=20)
    parser.add_argument("--pad-x-max", type=int, default=40)
    parser.add_argument("--pad-y-min", type=int, default=10)
    parser.add_argument("--pad-y-max", type=int, default=20)
    parser.add_argument("--min-width", type=int, default=48)
    parser.add_argument("--min-height", type=int, default=24)
    parser.add_argument("--max-text-region", type=int, default=0, help="0 keeps all valid text-region crops.")
    parser.add_argument("--random-ratio-to-text", type=float, default=1 / 3)
    parser.add_argument("--solid-ratio-to-text", type=float, default=1 / 6)
    parser.add_argument("--texture-ratio-to-text", type=float, default=1 / 6)
    parser.add_argument("--review-samples", type=int, default=600)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_id(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_") or "item"


def resolve_path(value: Any, workspace_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(workspace_root / path)
    text = str(value).replace("\\", "/")
    for marker in ("Meme_Dataset_Project/", "Misogyny_Dataset_Project/", "final_multilingual_meme_ocr_dataset/"):
        if marker in text:
            candidates.append(workspace_root / text[text.index(marker) :])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def record_split(record: dict[str, Any]) -> str:
    return str(record.get("final_split") or record.get("split") or "")


def source_id(record: dict[str, Any]) -> str:
    dataset = str(record.get("source_dataset") or record.get("dataset_source") or "dataset")
    image = Path(str(record.get("image") or record.get("source_image") or record.get("sample_id") or "image")).stem
    return f"{dataset}:{image}"


def parse_box(box: Any) -> tuple[int, int, int, int] | None:
    if not box:
        return None
    if isinstance(box, dict):
        if all(k in box for k in ("x", "y", "width", "height")):
            x = float(box["x"])
            y = float(box["y"])
            return (round(x), round(y), round(x + float(box["width"])), round(y + float(box["height"])))
        if all(k in box for k in ("left", "top", "right", "bottom")):
            return tuple(round(float(box[k])) for k in ("left", "top", "right", "bottom"))  # type: ignore[return-value]
    if isinstance(box, list) and len(box) >= 4:
        if all(isinstance(v, (int, float)) for v in box[:4]):
            x, y, w, h = [float(v) for v in box[:4]]
            return (round(x), round(y), round(x + w), round(y + h))
        if all(isinstance(v, list) and len(v) >= 2 for v in box):
            xs = [float(v[0]) for v in box]
            ys = [float(v[1]) for v in box]
            return (round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys)))
    return None


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    px = rng.randint(args.pad_x_min, args.pad_x_max)
    py = rng.randint(args.pad_y_min, args.pad_y_max)
    return clamp_box((x1 - px, y1 - py, x2 + px, y2 + py), width, height)


def box_size(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return box[2] - box[0], box[3] - box[1]


def intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def quality_ok(image: Image.Image, min_width: int, min_height: int) -> tuple[bool, dict[str, float]]:
    width, height = image.size
    if width < min_width or height < min_height:
        return False, {"width": width, "height": height, "mean": 0.0, "std": 0.0}
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    mean = float(stat.mean[0])
    std = float(stat.stddev[0])
    ok = 4.0 <= mean <= 251.0 and std >= 1.5
    return ok, {"width": width, "height": height, "mean": mean, "std": std}


def choose_bbox(line: dict[str, Any], args: argparse.Namespace) -> tuple[str, tuple[int, int, int, int]] | None:
    fields = [args.bbox_field] + [x.strip() for x in args.fallback_bbox_fields.split(",") if x.strip()]
    seen = set()
    for field in fields:
        if field in seen:
            continue
        seen.add(field)
        box = parse_box(line.get(field))
        if box:
            return field, box
    return None


def load_sources(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = json.loads(args.index_json.read_text(encoding="utf-8"))
    sources = []
    for record in records:
        if record_split(record) != "train":
            continue
        clean_path = (
            resolve_path(record.get("clean_image_abs"), args.workspace_root)
            or resolve_path(record.get("clean_image_workspace"), args.workspace_root)
            or resolve_path(record.get("clean_image"), args.workspace_root)
        )
        if clean_path is None:
            continue
        boxes = []
        for line in record.get("texts") or []:
            chosen = choose_bbox(line, args)
            if not chosen:
                continue
            field, box = chosen
            boxes.append({"field": field, "box": box, "line_index": int(line.get("line_index", len(boxes)))})
        if not boxes:
            continue
        sources.append(
            {
                "record": record,
                "clean_path": clean_path,
                "source_id": source_id(record),
                "boxes": boxes,
            }
        )
    return sources


def save_crop(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    out_path: Path,
    quality: int,
) -> tuple[bool, dict[str, float]]:
    crop = image.crop(crop_box).convert("RGB")
    ok, metrics = quality_ok(crop, min_width=1, min_height=1)
    if not ok:
        return False, metrics
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, quality=quality, optimize=True)
    return True, metrics


def make_manifest_row(
    bg_id: str,
    rel_path: Path,
    bg_type: str,
    source: dict[str, Any] | None,
    crop_box: tuple[int, int, int, int] | None,
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "id": bg_id,
        "image": rel_path.as_posix(),
        "background_type": bg_type,
        "width": int(metrics["width"]),
        "height": int(metrics["height"]),
        "luminance_mean": round(float(metrics["mean"]), 4),
        "luminance_std": round(float(metrics["std"]), 4),
    }
    if source is not None:
        record = source["record"]
        row.update(
            {
                "source_id": source["source_id"],
                "source_dataset": record.get("source_dataset") or record.get("dataset_source") or "",
                "source_image": record.get("source_image_workspace") or record.get("source_image_abs") or record.get("source_image") or "",
                "clean_image": str(source["clean_path"]),
            }
        )
    else:
        row.update({"source_id": "", "source_dataset": "procedural", "source_image": "", "clean_image": ""})
    if crop_box is not None:
        row["crop_box"] = {"x": crop_box[0], "y": crop_box[1], "width": crop_box[2] - crop_box[0], "height": crop_box[3] - crop_box[1]}
    else:
        row["crop_box"] = {}
    if extra:
        row.update(extra)
    return row


def build_text_region_backgrounds(
    sources: list[dict[str, Any]],
    output_dir: Path,
    rng: random.Random,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    shuffled_sources = list(sources)
    rng.shuffle(shuffled_sources)
    stop_after = args.max_text_region if args.max_text_region > 0 else None
    for source in shuffled_sources:
        try:
            with Image.open(source["clean_path"]) as img:
                img = img.convert("RGB")
                box_infos = list(source["boxes"])
                rng.shuffle(box_infos)
                for box_info in box_infos:
                    if stop_after is not None and len(rows) >= stop_after:
                        return rows
                    expanded = expand_box(box_info["box"], img.width, img.height, rng, args)
                    if expanded is None:
                        continue
                    w, h = box_size(expanded)
                    if w < args.min_width or h < args.min_height:
                        continue
                    next_idx = len(rows) + 1
                    rel = Path("meme_text_region") / f"meme_text_region_{next_idx:06d}.jpg"
                    ok, metrics = save_crop(img, expanded, output_dir / rel, args.jpeg_quality)
                    if not ok:
                        continue
                    rows.append(
                        make_manifest_row(
                            f"meme_text_region_{len(rows) + 1:06d}",
                            rel,
                            "meme_text_region",
                            source,
                            expanded,
                            metrics,
                            {"bbox_field": box_info["field"], "line_index": box_info["line_index"]},
                        )
                    )
                    if args.progress_every and len(rows) % args.progress_every == 0:
                        print(json.dumps({"stage": "meme_text_region", "rows": len(rows)}, ensure_ascii=False))
        except Exception:
            continue
    return rows


def sample_random_box(
    width: int,
    height: int,
    size_pool: list[tuple[int, int]],
    avoid_boxes: list[tuple[int, int, int, int]],
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[int, int, int, int] | None:
    if width < args.min_width or height < args.min_height:
        return None
    for _ in range(80):
        bw, bh = rng.choice(size_pool)
        scale = rng.uniform(0.85, 1.7)
        cw = min(width, max(args.min_width, round(bw * scale)))
        ch = min(height, max(args.min_height, round(bh * scale)))
        if cw >= width or ch >= height:
            continue
        x1 = rng.randint(0, width - cw)
        y1 = rng.randint(0, height - ch)
        cand = (x1, y1, x1 + cw, y1 + ch)
        area = cw * ch
        if all(intersection_area(cand, box) / max(1, area) <= 0.03 for box in avoid_boxes):
            return cand
    return None


def build_random_backgrounds(
    sources: list[dict[str, Any]],
    output_dir: Path,
    target: int,
    rng: random.Random,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    size_pool = []
    for source in sources:
        for item in source["boxes"]:
            w, h = box_size(item["box"])
            if w >= args.min_width and h >= args.min_height:
                size_pool.append((w, h))
    if not size_pool:
        size_pool = [(256, 96), (384, 96), (512, 128)]
    passes = 0
    while len(rows) < target and passes < 6:
        passes += 1
        shuffled_sources = list(sources)
        rng.shuffle(shuffled_sources)
        for source in shuffled_sources:
            if len(rows) >= target:
                break
            try:
                with Image.open(source["clean_path"]) as img:
                    img = img.convert("RGB")
                    avoid = []
                    for item in source["boxes"]:
                        expanded = clamp_box((item["box"][0] - 8, item["box"][1] - 8, item["box"][2] + 8, item["box"][3] + 8), img.width, img.height)
                        if expanded:
                            avoid.append(expanded)
                    # Later passes ask a little more from each image, but the
                    # first pass keeps visual-source diversity high.
                    tries_for_source = 1 if passes == 1 else 2
                    for _ in range(tries_for_source):
                        if len(rows) >= target:
                            break
                        crop_box = sample_random_box(img.width, img.height, size_pool, avoid, rng, args)
                        if crop_box is None:
                            continue
                        rel = Path("meme_random_nontext") / f"meme_random_nontext_{len(rows) + 1:06d}.jpg"
                        ok, metrics = save_crop(img, crop_box, output_dir / rel, args.jpeg_quality)
                        if not ok:
                            continue
                        rows.append(make_manifest_row(f"meme_random_nontext_{len(rows) + 1:06d}", rel, "meme_random_nontext", source, crop_box, metrics))
                        if args.progress_every and len(rows) % args.progress_every == 0:
                            print(json.dumps({"stage": "meme_random_nontext", "rows": len(rows), "target": target, "passes": passes}, ensure_ascii=False))
            except Exception:
                continue
    return rows


def size_from_text_rows(rows: list[dict[str, Any]], rng: random.Random) -> tuple[int, int]:
    if rows:
        row = rng.choice(rows)
        w = int(row["width"])
        h = int(row["height"])
    else:
        w, h = rng.choice([(256, 80), (384, 96), (512, 128), (640, 128)])
    w = max(96, min(900, round(w * rng.uniform(0.85, 1.25))))
    h = max(32, min(180, round(h * rng.uniform(0.85, 1.25))))
    return w, h


def build_solid_gradient(width: int, height: int, rng: random.Random) -> Image.Image:
    c1 = tuple(rng.randint(18, 238) for _ in range(3))
    c2 = tuple(max(0, min(255, c + rng.randint(-45, 45))) for c in c1)
    img = Image.new("RGB", (width, height), c1)
    pix = img.load()
    horizontal = rng.random() < 0.5
    denom = max(1, width - 1 if horizontal else height - 1)
    for y in range(height):
        for x in range(width):
            t = (x if horizontal else y) / denom
            pix[x, y] = tuple(round(c1[i] * (1 - t) + c2[i] * t) for i in range(3))
    if rng.random() < 0.45:
        draw = ImageDraw.Draw(img, "RGBA")
        for _ in range(rng.randint(1, 4)):
            cx = rng.randint(0, width)
            cy = rng.randint(0, height)
            r = rng.randint(max(16, min(width, height) // 5), max(24, min(width, height)))
            color = tuple(rng.randint(0, 255) for _ in range(3)) + (rng.randint(16, 45),)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.0, 4.0)))
    return img


def build_texture_noise(width: int, height: int, rng: random.Random) -> Image.Image:
    base = Image.effect_noise((width, height), rng.uniform(18, 55)).convert("L")
    base = base.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.2)))
    tint = tuple(rng.randint(45, 215) for _ in range(3))
    img = Image.merge("RGB", tuple(base.point(lambda p, c=c: max(0, min(255, int((p - 128) * rng.uniform(0.35, 0.9) + c)))) for c in tint))
    draw = ImageDraw.Draw(img, "RGBA")
    if rng.random() < 0.7:
        step = rng.randint(8, 28)
        color = tuple(rng.randint(0, 255) for _ in range(3)) + (rng.randint(8, 28),)
        for x in range(-height, width, step):
            draw.line((x, 0, x + height, height), fill=color, width=rng.randint(1, 3))
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))
    return img


def save_procedural(
    output_dir: Path,
    text_rows: list[dict[str, Any]],
    target: int,
    bg_type: str,
    rng: random.Random,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    subdir = "procedural_solid_gradient" if bg_type == "procedural_solid_gradient" else "procedural_texture_noise"
    builder = build_solid_gradient if bg_type == "procedural_solid_gradient" else build_texture_noise
    for idx in range(1, target + 1):
        w, h = size_from_text_rows(text_rows, rng)
        img = builder(w, h, rng)
        ok, metrics = quality_ok(img, args.min_width, args.min_height)
        if not ok:
            continue
        rel = Path(subdir) / f"{subdir}_{idx:06d}.jpg"
        (output_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_dir / rel, quality=args.jpeg_quality, optimize=True)
        rows.append(make_manifest_row(f"{subdir}_{idx:06d}", rel, bg_type, None, None, metrics, {"generator": subdir}))
    return rows


def write_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review_html(output_dir: Path, rows: list[dict[str, Any]], sample_count: int, rng: random.Random) -> None:
    sample = list(rows)
    rng.shuffle(sample)
    sample = sample[:sample_count]
    cards = []
    for row in sample:
        title = f"{row['id']} | {row['background_type']} | {row['width']}x{row['height']}"
        source = f"{row.get('source_dataset', '')} {row.get('source_id', '')}"
        cards.append(
            "<section class='card'>"
            f"<img src='{row['image']}' alt='{row['id']}'>"
            f"<div class='meta'><b>{title}</b><br>{source}<br>mean={row['luminance_mean']} std={row['luminance_std']}</div>"
            "</section>"
        )
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Background Library Review</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f5f3ee; color: #1f2933; }
    header { position: sticky; top: 0; background: #fffdf7; border-bottom: 1px solid #d8d3c8; padding: 12px 18px; }
    h1 { margin: 0 0 6px; font-size: 22px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; padding: 16px; }
    .card { background: white; border: 1px solid #d7d7d0; border-radius: 6px; padding: 10px; }
    img { width: 100%; height: 128px; object-fit: contain; background: #ebe7dc; border: 1px solid #e2ded5; }
    .meta { margin-top: 8px; font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }
  </style>
</head>
<body>
  <header>
    <h1>Background Library Review</h1>
    <div>Random sample for checking residual text and bad crops.</div>
  </header>
  <main class="grid">
"""
    html += "\n".join(cards)
    html += "\n  </main>\n</body>\n</html>\n"
    (output_dir / "review.html").write_text(html, encoding="utf-8")


def output_summary(args: argparse.Namespace, sources: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["background_type"] for row in rows)
    datasets = Counter(row.get("source_dataset", "") for row in rows if row.get("source_dataset") not in {"", "procedural"})
    summary = {
        "index_json": str(args.index_json),
        "output_dir": str(args.output_dir),
        "bbox_field": args.bbox_field,
        "seed": args.seed,
        "source_images": len(sources),
        "source_text_boxes": sum(len(source["boxes"]) for source in sources),
        "total_backgrounds": len(rows),
        "background_type_counts": dict(counts),
        "source_dataset_counts": dict(datasets),
        "ratios": {k: round(v / len(rows), 4) for k, v in counts.items()} if rows else {},
        "outputs": {
            "manifest": str(args.output_dir / "manifest.jsonl"),
            "summary": str(args.output_dir / "summary.json"),
            "review_html": str(args.output_dir / "review.html"),
        },
        "notes": [
            "Only train split clean images are used.",
            "meme_text_region crops are expanded from the selected text bbox field on LaMa-cleaned images.",
            "meme_random_nontext crops are sampled from cleaned train images while avoiding known text boxes.",
            "procedural_solid_gradient and procedural_texture_noise contain no source image.",
        ],
    }
    return summary


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.workspace_root = args.workspace_root.resolve()
    args.index_json = args.index_json.resolve()
    args.output_dir = args.output_dir.resolve()

    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists; pass --overwrite to replace it: {args.output_dir}")
        if "background_library" not in args.output_dir.name:
            raise SystemExit(f"Refusing to overwrite unexpected directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources(args)
    if not sources:
        raise SystemExit("No train clean images with boxes were found.")

    text_rows = build_text_region_backgrounds(sources, args.output_dir, rng, args)
    random_target = math.ceil(len(text_rows) * args.random_ratio_to_text)
    solid_target = math.ceil(len(text_rows) * args.solid_ratio_to_text)
    texture_target = math.ceil(len(text_rows) * args.texture_ratio_to_text)
    random_rows = build_random_backgrounds(sources, args.output_dir, random_target, rng, args)
    solid_rows = save_procedural(args.output_dir, text_rows, solid_target, "procedural_solid_gradient", rng, args)
    texture_rows = save_procedural(args.output_dir, text_rows, texture_target, "procedural_texture_noise", rng, args)
    rows = text_rows + random_rows + solid_rows + texture_rows

    write_manifest(args.output_dir, rows)
    write_review_html(args.output_dir, rows, args.review_samples, rng)
    summary = output_summary(args, sources, rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
