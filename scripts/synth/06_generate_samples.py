#!/usr/bin/env python3
"""Generate raw HTML/CSS/Selenium synthetic line images."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import re
import shutil
import time
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageChops, ImageStat
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


LANGS = ("zh", "ug", "kk")
KK_SPECIFIC = set("ӘәҒғҚқҢңӨөҰұҮүҺһІі")


def default_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    root = script_path.parents[2]
    return {
        "text_pool_dir": root / "02_corpus_preparation" / "mixed_text_pool_v2",
        "font_manifest": root / "03_synthetic_generation" / "font_library" / "final_check" / "font_manifest_final.json",
        "background_manifest": root / "03_synthetic_generation" / "background_library_v1" / "manifest.jsonl",
        "output_dir": root / "03_synthetic_generation" / "synthetic_smoke_v1",
    }


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-pool-dir", type=Path, default=defaults["text_pool_dir"])
    parser.add_argument("--font-manifest", type=Path, default=defaults["font_manifest"])
    parser.add_argument("--background-manifest", type=Path, default=defaults["background_manifest"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--per-lang", type=int, default=30)
    parser.add_argument("--languages", nargs="+", choices=LANGS, default=list(LANGS))
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--shard-id", default="shard_0000")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--text-shard-index", type=int, default=0)
    parser.add_argument("--text-shard-count", type=int, default=1)
    parser.add_argument("--text-shard-mode", choices=("stride", "global_window"), default="stride")
    parser.add_argument("--text-shard-size", type=int, default=0)
    parser.add_argument("--text-plan-seed", type=int, default=None)
    parser.add_argument("--max-text-occurrences", type=int, default=3)
    parser.add_argument("--style-profile", choices=("safe", "formal_diverse"), default="safe")
    parser.add_argument("--max-text-len-zh", type=int, default=30)
    parser.add_argument("--max-text-len-ugkk", type=int, default=45)
    parser.add_argument("--raw-height-min", type=int, default=64)
    parser.add_argument("--raw-height-max", type=int, default=128)
    parser.add_argument("--max-raw-width", type=int, default=1200)
    parser.add_argument("--min-font-size", type=int, default=24)
    parser.add_argument("--crop-pad-x", type=int, default=24)
    parser.add_argument("--crop-pad-y", type=int, default=12)
    parser.add_argument("--render-retries", type=int, default=8)
    parser.add_argument("--review-samples", type=int, default=120)
    parser.add_argument("--chrome-binary", default="")
    parser.add_argument("--driver-path", default="")
    parser.add_argument("--no-sandbox", action="store_true")
    parser.add_argument("--require-python-bidi", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[\u200b\u2060]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_text_pool(text_pool_dir: Path, language: str, args: argparse.Namespace) -> list[dict]:
    path = text_pool_dir / f"{language}_mixed_text_pool_v2.jsonl"
    rows = []
    for obj in read_jsonl(path):
        text = normalize_text(obj.get("normalized_text") or obj.get("text") or "")
        if not text:
            continue
        text_length = len(text.replace(" ", ""))
        max_len = args.max_text_len_zh if language == "zh" else args.max_text_len_ugkk
        if text_length > max_len:
            continue
        rows.append(
            {
                "language": language,
                "text": text,
                "source": obj.get("source") or "",
                "source_group": obj.get("source_group") or "",
                "domain": obj.get("domain") or "",
                "source_pool_id": obj.get("id") or obj.get("source_pool_id") or "",
                "text_length": text_length,
                "length_bucket": obj.get("length_bucket") or "",
            }
        )
    return rows


def resolve_font_path(path_text: str, font_manifest: Path, language: str, font_name: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    root = font_manifest.parents[1] / "raw_fonts" / language
    candidate = root / font_name
    if candidate.exists():
        return candidate
    matches = list(root.rglob(font_name))
    if matches:
        return matches[0]
    return path


def read_fonts(font_manifest: Path) -> dict[str, list[dict]]:
    data = json.loads(font_manifest.read_text(encoding="utf-8"))
    result: dict[str, list[dict]] = {}
    for language, rows in data.items():
        result[language] = []
        for row in rows:
            font_path = resolve_font_path(str(row.get("font_path") or ""), font_manifest, language, str(row.get("font") or ""))
            if not font_path.exists():
                continue
            item = dict(row)
            item["font_path_resolved"] = str(font_path)
            result[language].append(item)
    return result


def read_backgrounds(background_manifest: Path) -> list[dict]:
    base = background_manifest.parent
    rows = []
    for obj in read_jsonl(background_manifest):
        path = base / obj["image"]
        if not path.exists():
            continue
        item = dict(obj)
        item["path_resolved"] = str(path)
        item["aspect"] = float(item["width"]) / max(1.0, float(item["height"]))
        rows.append(item)
    return rows


def sample_texts(pools: dict[str, list[dict]], per_lang: int, rng: random.Random, languages: list[str]) -> list[dict]:
    targets = []
    for language in languages:
        rows = list(pools[language])
        if len(rows) < per_lang:
            raise SystemExit(f"Not enough {language} text rows: need {per_lang}, got {len(rows)}")
        targets.extend(rng.sample(rows, per_lang))
    rng.shuffle(targets)
    return targets


def shuffled_text_queues(
    pools: dict[str, list[dict]],
    per_lang: int,
    rng: random.Random,
    languages: list[str],
) -> dict[str, list[dict]]:
    queues = {}
    for language in languages:
        rows = list(pools[language])
        if len(rows) < per_lang:
            raise SystemExit(f"Not enough {language} text rows: need {per_lang}, got {len(rows)}")
        rng.shuffle(rows)
        queues[language] = rows
    return queues


def apply_text_sharding(
    pools: dict[str, list[dict]],
    shard_index: int,
    shard_count: int,
    mode: str = "stride",
    shard_size: int = 0,
    plan_seed: int = 0,
    max_occurrences: int = 3,
) -> dict[str, list[dict]]:
    if shard_count < 1:
        raise SystemExit(f"--text-shard-count must be >= 1, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise SystemExit(f"--text-shard-index must be in [0, {shard_count - 1}], got {shard_index}")
    if mode == "global_window" and shard_size < 1:
        raise SystemExit(f"--text-shard-size must be >= 1 in global_window mode, got {shard_size}")
    if max_occurrences < 1:
        raise SystemExit(f"--max-text-occurrences must be >= 1, got {max_occurrences}")

    sharded = {}
    for language, rows in pools.items():
        ordered = sorted(
            rows,
            key=lambda row: (
                str(row.get("source_pool_id") or ""),
                str(row.get("source") or ""),
                str(row.get("domain") or ""),
                str(row.get("text") or ""),
            ),
        )
        if mode == "stride":
            sharded[language] = ordered[shard_index::shard_count] if shard_count > 1 else ordered
            continue

        if shard_size > len(ordered):
            raise SystemExit(
                f"Global text shard for {language} needs {shard_size} distinct candidates, "
                f"but the filtered pool has only {len(ordered)}"
            )
        total_plan_slots = shard_count * shard_size
        required_occurrences = math.ceil(total_plan_slots / len(ordered))
        if required_occurrences > max_occurrences:
            raise SystemExit(
                f"Global text plan would use some {language} texts {required_occurrences} times "
                f"({total_plan_slots} slots / {len(ordered)} texts), exceeding "
                f"--max-text-occurrences={max_occurrences}. Enlarge the text pool or reduce the plan."
            )

        language_seed = int.from_bytes(
            hashlib.sha256(f"{plan_seed}:{language}".encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        )
        random.Random(language_seed).shuffle(ordered)
        start = shard_index * shard_size
        selected = []
        for offset in range(shard_size):
            plan_position = start + offset
            pool_index = plan_position % len(ordered)
            item = dict(ordered[pool_index])
            item["_text_plan_position"] = plan_position
            item["_text_plan_pool_index"] = pool_index
            item["_text_plan_occurrence"] = plan_position // len(ordered) + 1
            item["_text_plan_pool_size"] = len(ordered)
            selected.append(item)
        sharded[language] = selected
    return sharded


def weighted_background_type(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.60:
        return "meme_text_region"
    if roll < 0.80:
        return "meme_random_nontext"
    if roll < 0.90:
        return "procedural_solid_gradient"
    return "procedural_texture_noise"


def target_width_for_text(text: str, language: str) -> int:
    n = len(text.replace(" ", ""))
    if language == "zh":
        return min(1120, max(180, n * 38))
    return min(1120, max(220, n * 22))


def background_allowed(background: dict, language: str) -> bool:
    bg_type = str(background.get("background_type", ""))
    mean = float(background.get("luminance_mean", 128))
    if bg_type.startswith("meme_"):
        min_mean = 70.0 if language == "ug" else 55.0
        if mean < min_mean:
            return False
    if language == "ug" and mean < 35.0:
        return False
    return True


def choose_background(backgrounds: list[dict], by_type: dict[str, list[dict]], text: str, language: str, rng: random.Random) -> dict:
    desired_type = weighted_background_type(rng)
    allowed_backgrounds = [b for b in backgrounds if background_allowed(b, language)] or backgrounds
    pool = by_type.get(desired_type) or allowed_backgrounds
    pool = [b for b in pool if background_allowed(b, language)] or allowed_backgrounds
    min_width = target_width_for_text(text, language)
    min_height = 48 if len(text.replace(" ", "")) < 24 else 56
    candidates = [b for b in pool if int(b["width"]) >= min_width and int(b["height"]) >= min_height]
    if not candidates:
        candidates = [b for b in allowed_backgrounds if int(b["width"]) >= min_width * 0.75 and int(b["height"]) >= min_height]
    if not candidates:
        candidates = allowed_backgrounds
    return rng.choice(candidates)


def choose_font(
    fonts: dict[str, list[dict]],
    language: str,
    rng: random.Random,
    style_profile: str = "safe",
) -> dict:
    pool = fonts.get(language) or []
    if not pool:
        raise SystemExit(f"No usable fonts for language={language}")
    diverse = style_profile == "formal_diverse"
    if not diverse and language == "ug":
        safe_pool = []
        for font in pool:
            name = f"{font.get('font','')} {font.get('font_name','')}".lower()
            if "kufi" in name:
                continue
            safe_pool.append(font)
        if safe_pool:
            pool = safe_pool
    if not diverse and language == "kk":
        safe_pool = []
        for font in pool:
            name = f"{font.get('font','')} {font.get('font_name','')}".lower()
            if "pt serif" in name:
                continue
            safe_pool.append(font)
        if safe_pool:
            pool = safe_pool

    weighted = []
    for font in pool:
        name = f"{font.get('font','')} {font.get('font_name','')}".lower()
        weight = 1
        if language == "ug":
            if any(key in name for key in ("naskh", "scheherazade")):
                weight = 7 if diverse else 8
            elif "sans arabic" in name:
                weight = 5 if diverse else 4
            elif "amiri" in name:
                weight = 4 if diverse else 3
            elif "plex" in name:
                weight = 3 if diverse else 2
            elif "kufi" in name:
                weight = 2 if diverse else 0
        elif language == "zh":
            if "bold" in name:
                weight = 6
            elif "black" in name or "heavy" in name:
                weight = 4 if diverse else 3
            else:
                weight = 5 if diverse else 6
        elif language == "kk":
            if any(key in name for key in ("noto sans", "dejavu sans", "liberation sans", "pt sans")):
                weight = 7 if diverse else 6
            elif any(key in name for key in ("noto serif", "liberation serif")):
                weight = 4 if diverse else 6
            if "mono" in name:
                weight = 2 if diverse else 3
            if "display" in name or "pt serif" in name:
                weight = 1 if diverse else weight
        if weight > 0:
            weighted.extend([font] * weight)
    return rng.choice(weighted or pool)


def luminance_to_rgb(value: int) -> tuple[int, int, int]:
    value = max(0, min(255, value))
    return value, value, value


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def choose_colors(background: dict, rng: random.Random, language: str = "") -> tuple[str, str, str]:
    mean = float(background.get("luminance_mean", 128))
    bg_type = str(background.get("background_type", ""))
    readable_pairs = [
        ((255, 255, 255), (0, 0, 0)),
        ((255, 242, 70), (0, 0, 0)),
        ((70, 220, 255), (0, 0, 0)),
        ((20, 20, 20), (255, 255, 255)),
    ]
    if language == "ug":
        contrast = "high"
        fill, stroke = rng.choice(readable_pairs[:2])
        return rgb_to_hex(fill), rgb_to_hex(stroke), contrast
    roll = rng.random()
    if roll < 0.75:
        contrast = "high"
        if bg_type.startswith("meme_"):
            fill, stroke = rng.choice(readable_pairs[:3])
        else:
            fill = (245, 245, 245) if mean < 132 else (20, 20, 20)
            stroke = (0, 0, 0) if sum(fill) / 3 > 130 else (255, 255, 255)
    else:
        contrast = "medium"
        if bg_type.startswith("meme_"):
            fill = rng.choice([(255, 255, 255), (255, 232, 40), (70, 220, 255)])
        else:
            palette = [(255, 232, 40), (255, 72, 72), (55, 220, 255), (250, 250, 250), (30, 30, 30)]
            fill = rng.choice(palette)
            if abs(sum(fill) / 3 - mean) < 50:
                fill = (245, 245, 245) if mean < 128 else (20, 20, 20)
        stroke = (0, 0, 0) if sum(fill) / 3 > 130 else (255, 255, 255)
    return rgb_to_hex(fill), rgb_to_hex(stroke), contrast


def base_font_size(text: str, language: str, bg: dict, rng: random.Random) -> int:
    n = len(text.replace(" ", ""))
    h = int(bg["height"])
    if language == "zh":
        base = 50 if n <= 4 else 54 if n <= 8 else 48 if n <= 16 else 42 if n <= 28 else 34
    elif language == "ug":
        base = 66 if n <= 12 else 58 if n <= 24 else 48 if n <= 38 else 40
    else:
        base = 62 if n <= 12 else 52 if n <= 24 else 42 if n <= 38 else 34
    base = min(base, max(22, round(h * rng.uniform(0.55, 0.82))))
    minimum = 30 if language == "ug" else 28
    return max(minimum, base + rng.randint(-4, 4))


def minimum_final_font_size(language: str) -> int:
    if language == "ug":
        return 30
    if language == "kk":
        return 28
    return 28


def choose_style_safe(row: dict, font: dict, bg: dict, rng: random.Random) -> dict:
    language = row["language"]
    text = row["text"]
    fill, stroke_color, contrast = choose_colors(bg, rng, language)
    meme_bg = str(bg.get("background_type", "")).startswith("meme_")
    no_effect = (rng.random() < 0.20) and not meme_bg
    stroke_width = 0 if no_effect or rng.random() >= 0.60 else rng.choice([1, 1, 2, 2, 3])
    shadow = "none"
    shadow_flag = False
    glow = False
    if not no_effect and rng.random() < 0.40:
        shadow_flag = True
        shadow = rng.choice(
            [
                "2px 2px 0 rgba(0,0,0,.75)",
                "2px 2px 4px rgba(0,0,0,.65)",
                "0 2px 5px rgba(0,0,0,.55)",
                "1px 1px 0 rgba(255,255,255,.75)",
            ]
        )
    if not no_effect and rng.random() < 0.09:
        glow = True
        glow_color = "rgba(255,255,255,.85)" if fill != "#ffffff" else "rgba(0,0,0,.65)"
        shadow = f"{shadow}, 0 0 8px {glow_color}" if shadow != "none" else f"0 0 8px {glow_color}"
    double_effect = (not no_effect) and rng.random() < 0.15
    if contrast in {"medium", "low"}:
        no_effect = False
        if stroke_width <= 0 and not shadow_flag:
            stroke_width = 1
        if contrast == "low" and not shadow_flag:
            shadow_flag = True
            shadow = "1px 1px 3px rgba(0,0,0,.55)" if stroke_color == "#000000" else "1px 1px 3px rgba(255,255,255,.60)"
    if meme_bg:
        stroke_width = max(stroke_width, 2)
        if not shadow_flag and rng.random() < 0.50:
            shadow_flag = True
            shadow = "2px 2px 3px rgba(0,0,0,.65)" if stroke_color == "#000000" else "2px 2px 3px rgba(255,255,255,.65)"
    if stroke_width <= 0:
        stroke_width = 1
        no_effect = False
    if language == "kk":
        stroke_width = max(stroke_width, 1)
    if language == "ug":
        no_effect = False
        stroke_width = rng.choice([1.0, 1.0, 1.2])
        shadow_flag = True
        shadow = "0 0 3px rgba(255,255,255,.42), 1px 1px 1px rgba(0,0,0,.38)"
        glow = False
        double_effect = False
    opacity = 1.0
    if language == "ug":
        rotate = rng.uniform(-1.0, 1.0) if rng.random() < 0.10 else 0.0
        skew = 0.0
        font_weight = rng.choice(["400", "500", "600"])
    else:
        rotate = rng.uniform(-2.2, 2.2) if (not no_effect and rng.random() < 0.20) else 0.0
        skew = rng.uniform(-5.0, 5.0) if (not no_effect and rng.random() < 0.10) else 0.0
        font_weight = rng.choice(["400", "500", "600"] if meme_bg else ["400", "500", "600", "700"])
    letter_spacing = 0.0
    margin_x = rng.randint(8, 28)
    margin_y = rng.randint(4, 16)
    return {
        "font_file": font["font"],
        "font_name": font.get("font_name") or font["font"],
        "font_path": font["font_path_resolved"],
        "font_size": base_font_size(text, language, bg, rng),
        "font_weight": font_weight,
        "font_style": "italic" if language != "ug" and rng.random() < 0.04 else "normal",
        "fill_color": fill,
        "stroke_color": stroke_color,
        "stroke_width": stroke_width,
        "shadow": shadow,
        "shadow_enabled": shadow_flag,
        "glow": glow,
        "double_effect": double_effect,
        "opacity": round(opacity, 4),
        "rotation": round(rotate, 4),
        "skew": round(skew, 4),
        "letter_spacing": letter_spacing,
        "margin_x": margin_x,
        "margin_y": margin_y,
        "contrast_bucket": contrast,
        "no_special_effect": no_effect,
    }


def choose_meme_style(language: str, rng: random.Random) -> str:
    if language == "ug":
        weighted = [
            ("outline", 30),
            ("shadow", 15),
            ("outline_shadow", 30),
            ("double_shadow", 7),
            ("glow", 4),
            ("translucent", 5),
            ("caption_box", 9),
        ]
    else:
        weighted = [
            ("plain", 12),
            ("outline", 24),
            ("shadow", 18),
            ("outline_shadow", 22),
            ("double_shadow", 8),
            ("glow", 6),
            ("translucent", 5),
            ("caption_box", 5),
        ]
    target = rng.uniform(0, sum(weight for _, weight in weighted))
    cumulative = 0.0
    for name, weight in weighted:
        cumulative += weight
        if target <= cumulative:
            return name
    return weighted[-1][0]


def color_luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def choose_colors_diverse(background: dict, rng: random.Random, language: str) -> tuple[str, str, str]:
    mean = float(background.get("luminance_mean", 128))
    roll = rng.random()
    if language == "ug":
        contrast = "high" if roll < 0.70 else "medium" if roll < 0.95 else "low"
    else:
        contrast = "high" if roll < 0.60 else "medium" if roll < 0.90 else "low"

    bright = [(255, 255, 255), (255, 232, 55), (80, 220, 255), (255, 145, 90)]
    dark = [(12, 12, 12), (92, 20, 28), (20, 42, 105), (55, 28, 92)]
    vivid = [(255, 72, 72), (255, 190, 40), (60, 210, 245), (105, 235, 120), (245, 245, 245)]

    if language == "ug":
        # Never use a black Uyghur fill. Thin Arabic strokes remain readable
        # through a bright fill plus a dark outline/shadow.
        if contrast == "high":
            fill = rng.choice(bright[:3])
        elif contrast == "medium":
            fill = rng.choice(bright[1:] + vivid[:3])
        else:
            near = max(145, min(225, round(mean + (-18 if mean > 175 else 28))))
            fill = (near, max(120, near - 15), min(255, near + 12))
        stroke = (8, 8, 8)
        return rgb_to_hex(fill), rgb_to_hex(stroke), contrast

    if contrast == "high":
        if mean < 108:
            fill = rng.choice(bright)
        elif mean > 170:
            fill = rng.choice(dark)
        else:
            fill = rng.choice(bright[:3] + dark[:2])
    elif contrast == "medium":
        fill = rng.choice(vivid + dark[1:])
    else:
        near = max(28, min(228, round(mean + rng.choice([-24, -18, 18, 24]))))
        fill = (near, near, near)
    stroke = (8, 8, 8) if color_luminance(fill) >= 132 else (250, 250, 250)
    return rgb_to_hex(fill), rgb_to_hex(stroke), contrast


def choose_style_diverse(row: dict, font: dict, bg: dict, rng: random.Random) -> dict:
    language = row["language"]
    text = row["text"]
    meme_style = choose_meme_style(language, rng)
    fill, stroke_color, contrast = choose_colors_diverse(bg, rng, language)
    meme_bg = str(bg.get("background_type", "")).startswith("meme_")

    shadow = "none"
    shadow_flag = False
    glow = meme_style == "glow"
    double_effect = meme_style == "double_shadow"
    opacity = 1.0
    backplate_color = "transparent"
    backplate_padding_x = 0
    backplate_padding_y = 0
    backplate_radius = 0

    if language == "ug":
        stroke_width = rng.choice([0.8, 1.0, 1.0, 1.2, 1.2])
    else:
        stroke_width = rng.choice([1.0, 1.5, 2.0, 2.0, 2.5])

    if meme_style == "plain":
        stroke_width = 1.0 if meme_bg else 0.0
    elif meme_style == "outline":
        pass
    elif meme_style == "shadow":
        stroke_width = rng.choice([0.0, 0.8, 1.0]) if language != "ug" else rng.choice([0.8, 1.0])
        shadow_flag = True
        shadow = "2px 2px 3px rgba(0,0,0,.72), 0 1px 1px rgba(255,255,255,.28)"
    elif meme_style == "outline_shadow":
        shadow_flag = True
        shadow = "2px 2px 2px rgba(0,0,0,.72), 0 0 2px rgba(255,255,255,.30)"
    elif meme_style == "double_shadow":
        shadow_flag = True
        shadow = "2px 2px 0 rgba(0,0,0,.78), -1px -1px 0 rgba(255,255,255,.62), 0 0 3px rgba(0,0,0,.45)"
    elif meme_style == "glow":
        shadow_flag = True
        glow_color = "rgba(255,245,150,.82)" if language == "ug" else "rgba(255,255,255,.78)"
        shadow = f"0 0 2px {glow_color}, 0 0 6px {glow_color}, 1px 1px 2px rgba(0,0,0,.62)"
    elif meme_style == "translucent":
        shadow_flag = True
        opacity = rng.uniform(0.84 if language == "ug" else 0.80, 0.94)
        shadow = "1px 1px 3px rgba(0,0,0,.68)"
    elif meme_style == "caption_box":
        stroke_width = 0.8 if language == "ug" else rng.choice([0.0, 0.8, 1.0])
        fill_luma = int(fill[1:3], 16) * 0.2126 + int(fill[3:5], 16) * 0.7152 + int(fill[5:7], 16) * 0.0722
        backplate_color = "rgba(0,0,0,.58)" if language == "ug" or fill_luma > 132 else "rgba(255,255,255,.66)"
        backplate_padding_x = rng.randint(7, 14)
        backplate_padding_y = rng.randint(3, 7)
        backplate_radius = rng.choice([0, 2, 4])

    if contrast == "low":
        stroke_width = max(stroke_width, 1.0 if language == "ug" else 1.5)
        if not shadow_flag and meme_style != "caption_box":
            shadow_flag = True
            shadow = "1px 1px 3px rgba(0,0,0,.68)"

    if language == "ug":
        # Preserve joining and thin glyph interiors while allowing real meme effects.
        stroke_width = min(1.4, max(0.8, stroke_width))
        rotate = rng.uniform(-1.5, 1.5) if rng.random() < 0.15 else 0.0
        skew = rng.uniform(-2.0, 2.0) if rng.random() < 0.05 else 0.0
        font_weight = rng.choice(["500", "500", "600", "600", "700"])
        letter_spacing = 0.0
    else:
        rotate = rng.uniform(-3.0, 3.0) if rng.random() < 0.25 else 0.0
        skew = rng.uniform(-5.0, 5.0) if rng.random() < 0.12 else 0.0
        font_weight = rng.choice(["400", "500", "600", "600", "700", "700"])
        letter_spacing = round(rng.uniform(-0.25, 0.65), 3) if rng.random() < 0.12 else 0.0

    text_length = len(text.replace(" ", ""))
    if text_length > 32 or rng.random() < 0.65:
        position_x = 50.0
        position_y = 50.0
    else:
        position_x = rng.uniform(43.0, 57.0)
        position_y = rng.uniform(40.0, 60.0)

    return {
        "font_file": font["font"],
        "font_name": font.get("font_name") or font["font"],
        "font_path": font["font_path_resolved"],
        "font_size": base_font_size(text, language, bg, rng),
        "font_weight": font_weight,
        "font_style": "italic" if language != "ug" and rng.random() < 0.05 else "normal",
        "fill_color": fill,
        "stroke_color": stroke_color,
        "stroke_width": round(stroke_width, 3),
        "shadow": shadow,
        "shadow_enabled": shadow_flag,
        "glow": glow,
        "double_effect": double_effect,
        "opacity": round(opacity, 4),
        "rotation": round(rotate, 4),
        "skew": round(skew, 4),
        "letter_spacing": letter_spacing,
        "margin_x": rng.randint(10, 32),
        "margin_y": rng.randint(6, 18),
        "position_x": round(position_x, 4),
        "position_y": round(position_y, 4),
        "backplate_color": backplate_color,
        "backplate_padding_x": backplate_padding_x,
        "backplate_padding_y": backplate_padding_y,
        "backplate_radius": backplate_radius,
        "contrast_bucket": contrast,
        "meme_style": meme_style,
        "no_special_effect": meme_style == "plain",
    }


def choose_style(
    row: dict,
    font: dict,
    bg: dict,
    rng: random.Random,
    style_profile: str = "safe",
) -> dict:
    if style_profile == "formal_diverse":
        return choose_style_diverse(row, font, bg, rng)
    style = choose_style_safe(row, font, bg, rng)
    style.update(
        {
            "position_x": 50.0,
            "position_y": 50.0,
            "backplate_color": "transparent",
            "backplate_padding_x": 0,
            "backplate_padding_y": 0,
            "backplate_radius": 0,
            "meme_style": "safe_legacy",
        }
    )
    return style

def file_url(path: Path) -> str:
    return path.resolve().as_uri()


def build_driver(args: argparse.Namespace):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1400,700")
    options.add_argument("--allow-file-access-from-files")
    options.add_argument("--disable-web-security")
    if args.no_sandbox:
        options.add_argument("--no-sandbox")
    if args.chrome_binary:
        options.binary_location = args.chrome_binary
    service = Service(args.driver_path) if args.driver_path else Service()
    return webdriver.Chrome(service=service, options=options)


def write_template(path: Path) -> None:
    path.write_text(
        """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style id="dynamicFont"></style>
  <style>
    html, body { margin: 0; padding: 0; background: #808080; }
    body { min-width: 1280px; min-height: 420px; display: flex; align-items: center; justify-content: center; }
    #stage {
      position: relative;
      overflow: hidden;
      background-size: 100% 100%;
      background-position: center;
      background-repeat: no-repeat;
    }
    #text {
      position: absolute;
      left: 50%;
      top: 50%;
      max-width: calc(100% - var(--margin-x) * 2);
      max-height: calc(100% - var(--margin-y) * 2);
      white-space: nowrap;
      line-height: 1.06;
      text-align: center;
      transform-origin: center center;
      text-rendering: geometricPrecision;
      -webkit-font-smoothing: antialiased;
      font-kerning: normal;
      font-variant-ligatures: common-ligatures contextual;
      font-feature-settings: "rlig" 1, "calt" 1, "liga" 1, "kern" 1;
      unicode-bidi: isolate;
    }
  </style>
</head>
<body>
  <div id="stage"><div id="text"></div></div>
  <script>
    window.renderSample = async function(cfg) {
      const fontStyle = document.getElementById("dynamicFont");
      fontStyle.textContent = "@font-face{font-family:'SampleFont';src:url('" + cfg.fontUrl + "');}";
      const stage = document.getElementById("stage");
      const text = document.getElementById("text");
      stage.style.width = cfg.stageWidth + "px";
      stage.style.height = cfg.stageHeight + "px";
      stage.style.backgroundImage = "url('" + cfg.backgroundUrl + "')";
      stage.style.setProperty("--margin-x", cfg.marginX + "px");
      stage.style.setProperty("--margin-y", cfg.marginY + "px");
      text.textContent = cfg.text;
      text.setAttribute("lang", cfg.language);
      text.setAttribute("dir", cfg.direction);
      text.style.fontFamily = "'SampleFont'";
      text.style.left = cfg.positionX + "%";
      text.style.top = cfg.positionY + "%";
      text.style.fontSize = cfg.fontSize + "px";
      text.style.fontWeight = cfg.fontWeight;
      text.style.fontStyle = cfg.fontStyle;
      text.style.color = cfg.fillColor;
      text.style.webkitTextStroke = cfg.strokeWidth + "px " + cfg.strokeColor;
      text.style.textShadow = cfg.shadow;
      text.style.opacity = cfg.opacity;
      text.style.letterSpacing = cfg.letterSpacing + "px";
      text.style.backgroundColor = cfg.backplateColor;
      text.style.padding = cfg.backplatePaddingY + "px " + cfg.backplatePaddingX + "px";
      text.style.borderRadius = cfg.backplateRadius + "px";
      text.style.transform = "translate(-50%, -50%) rotate(" + cfg.rotation + "deg) skewX(" + cfg.skew + "deg)";
      await document.fonts.ready;
      await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      let fontSize = cfg.fontSize;
      const effectPadX = Math.max(18, cfg.strokeWidth * 5 + (cfg.shadow === "none" ? 0 : 12));
      const effectPadY = Math.max(12, cfg.strokeWidth * 4 + (cfg.shadow === "none" ? 0 : 10));
      const maxWidth = cfg.stageWidth - cfg.marginX * 2 - effectPadX;
      const maxHeight = cfg.stageHeight - cfg.marginY * 2 - effectPadY;
      for (let i = 0; i < 24; i++) {
        const rect = text.getBoundingClientRect();
        if ((rect.width <= maxWidth && rect.height <= maxHeight) || fontSize <= cfg.minFontSize) break;
        fontSize = Math.max(cfg.minFontSize, Math.floor(fontSize * 0.92));
        text.style.fontSize = fontSize + "px";
        await new Promise(r => requestAnimationFrame(r));
      }
      await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      const rect = text.getBoundingClientRect();
      const stageRect = stage.getBoundingClientRect();
      return {
        finalFontSize: fontSize,
        textBox: {x: rect.x - stageRect.x, y: rect.y - stageRect.y, width: rect.width, height: rect.height},
        stageBox: {width: stageRect.width, height: stageRect.height},
        clipped: rect.x < stageRect.x + effectPadX / 2 || rect.y < stageRect.y + effectPadY / 2 || rect.right > stageRect.right - effectPadX / 2 || rect.bottom > stageRect.bottom - effectPadY / 2
      };
    }
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def make_sample_id(language: str, idx: int) -> str:
    return f"{language}_{idx:06d}"


def ug_visual_label(text: str, require_python_bidi: bool = False) -> tuple[str, str]:
    try:
        from bidi.algorithm import get_display  # type: ignore

        return get_display(text, base_dir="R"), "python_bidi"
    except Exception as exc:
        if require_python_bidi:
            raise RuntimeError("python-bidi is required for formal Uyghur U2 labels") from exc
        return text, "unavailable_keep_logical"


def clamp_box(box: dict, width: int, height: int, pad: int = 0) -> tuple[int, int, int, int]:
    x0 = max(0, int(math.floor(float(box.get("x", 0)))) - pad)
    y0 = max(0, int(math.floor(float(box.get("y", 0)))) - pad)
    x1 = min(width, int(math.ceil(float(box.get("x", 0)) + float(box.get("width", 0)))) + pad)
    y1 = min(height, int(math.ceil(float(box.get("y", 0)) + float(box.get("height", 0)))) + pad)
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def diff_changed_ratio(diff_img: Image.Image, threshold: int = 24) -> tuple[float, float, float]:
    hist = diff_img.histogram()
    total = max(1, diff_img.width * diff_img.height)
    changed = sum(hist[threshold:])
    strong = sum(hist[50:])
    weighted = sum(i * count for i, count in enumerate(hist))
    changed_weighted = sum(i * hist[i] for i in range(threshold, 256))
    changed_ratio = changed / total
    strong_ratio = strong / total
    changed_mean_delta = changed_weighted / max(1, changed)
    mean_delta = weighted / total
    return changed_ratio, strong_ratio, max(changed_mean_delta, mean_delta)


def evaluate_render_quality(raw_path: Path, bg: dict, render_info: dict, language: str) -> dict:
    img = Image.open(raw_path).convert("RGB")
    bg_img = Image.open(bg["path_resolved"]).convert("RGB").resize(img.size, Image.Resampling.BICUBIC)
    diff = ImageChops.difference(img, bg_img).convert("L")
    x0, y0, x1, y1 = clamp_box(render_info.get("textBox", {}), img.width, img.height, pad=3)
    crop = diff.crop((x0, y0, x1, y1))
    changed_ratio, strong_ratio, changed_mean_delta = diff_changed_ratio(crop)
    raw_luma_std = float(ImageStat.Stat(img.crop((x0, y0, x1, y1)).convert("L")).stddev[0])

    segment_ratios = []
    segment_strong_ratios = []
    if x1 - x0 >= 36:
        segment_count = 5
        for idx in range(segment_count):
            sx0 = x0 + (x1 - x0) * idx // segment_count
            sx1 = x0 + (x1 - x0) * (idx + 1) // segment_count
            seg = diff.crop((sx0, y0, max(sx0 + 1, sx1), y1))
            seg_changed, seg_strong, _ = diff_changed_ratio(seg)
            segment_ratios.append(seg_changed)
            segment_strong_ratios.append(seg_strong)
    min_segment_ratio = min(segment_ratios) if segment_ratios else changed_ratio
    min_segment_strong_ratio = min(segment_strong_ratios) if segment_strong_ratios else strong_ratio

    min_changed = {"zh": 0.022, "ug": 0.012, "kk": 0.020}.get(language, 0.018)
    min_strong = {"zh": 0.010, "ug": 0.005, "kk": 0.009}.get(language, 0.007)
    min_delta = {"zh": 38.0, "ug": 32.0, "kk": 36.0}.get(language, 34.0)
    if str(bg.get("background_type", "")).startswith("meme_"):
        min_delta += 6.0
    min_std = {"zh": 10.0, "ug": 8.0, "kk": 9.0}.get(language, 9.0)

    reasons = []
    if changed_ratio < min_changed:
        reasons.append(f"changed_ratio_lt_{min_changed:.3f}")
    if strong_ratio < min_strong:
        reasons.append(f"strong_ratio_lt_{min_strong:.3f}")
    if changed_mean_delta < min_delta:
        reasons.append(f"changed_delta_lt_{min_delta:.1f}")
    if raw_luma_std < min_std:
        reasons.append(f"text_region_std_lt_{min_std:.1f}")
    if (x1 - x0) >= 120 and min_segment_ratio < min_changed * 0.65:
        reasons.append("weak_local_segment")
    if (x1 - x0) >= 120 and min_segment_strong_ratio < min_strong * 0.60:
        reasons.append("weak_local_strong_segment")

    return {
        "quality_pass": not reasons,
        "quality_reasons": reasons,
        "quality_changed_ratio": round(changed_ratio, 6),
        "quality_strong_ratio": round(strong_ratio, 6),
        "quality_changed_mean_delta": round(changed_mean_delta, 4),
        "quality_text_region_std": round(raw_luma_std, 4),
        "quality_min_segment_ratio": round(min_segment_ratio, 6),
        "quality_min_segment_strong_ratio": round(min_segment_strong_ratio, 6),
    }


def crop_rendered_line(raw_path: Path, render_info: dict, args: argparse.Namespace) -> dict:
    img = Image.open(raw_path).convert("RGB")
    text_box = render_info.get("textBox", {})
    font_size = int(render_info.get("finalFontSize") or args.min_font_size)
    pad_x = max(args.crop_pad_x, round(font_size * 0.45))
    pad_y = max(args.crop_pad_y, round(font_size * 0.25))
    x0, y0, x1, y1 = clamp_box(text_box, img.width, img.height, pad=0)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(img.width, x1 + pad_x)
    y1 = min(img.height, y1 + pad_y)
    cropped = img.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)))
    cropped.save(raw_path)
    return {
        "crop_box": {"x": x0, "y": y0, "width": cropped.width, "height": cropped.height},
        "raw_uncropped_width": img.width,
        "raw_uncropped_height": img.height,
    }


def render_one(driver, template_url: str, row: dict, font: dict, bg: dict, style: dict, raw_path: Path, args: argparse.Namespace) -> dict:
    driver.get(template_url)
    text_len = len(str(row["text"]).replace(" ", ""))
    stage_w = min(args.max_raw_width, max(96, int(bg["width"])))
    required_h = 80
    if text_len > 28:
        required_h = 96
    if text_len > 40:
        required_h = 112
    stage_h = max(args.raw_height_min, min(args.raw_height_max, max(int(bg["height"]), required_h)))
    cfg = {
        "text": row["text"],
        "language": row["language"],
        "direction": "rtl" if row["language"] == "ug" else "ltr",
        "fontUrl": file_url(Path(style["font_path"])),
        "backgroundUrl": file_url(Path(bg["path_resolved"])),
        "stageWidth": stage_w,
        "stageHeight": stage_h,
        "fontSize": style["font_size"],
        "minFontSize": args.min_font_size,
        "fontWeight": style["font_weight"],
        "fontStyle": style["font_style"],
        "fillColor": style["fill_color"],
        "strokeColor": style["stroke_color"],
        "strokeWidth": style["stroke_width"],
        "shadow": style["shadow"],
        "opacity": style["opacity"],
        "letterSpacing": style["letter_spacing"],
        "rotation": style["rotation"],
        "skew": style["skew"],
        "positionX": style["position_x"],
        "positionY": style["position_y"],
        "backplateColor": style["backplate_color"],
        "backplatePaddingX": style["backplate_padding_x"],
        "backplatePaddingY": style["backplate_padding_y"],
        "backplateRadius": style["backplate_radius"],
        "marginX": style["margin_x"],
        "marginY": style["margin_y"],
    }
    info = driver.execute_async_script(
        """
        const cfg = arguments[0];
        const done = arguments[arguments.length - 1];
        window.renderSample(cfg).then(done).catch(err => done({error: String(err)}));
        """,
        cfg,
    )
    if info.get("error"):
        raise RuntimeError(info["error"])
    if info.get("clipped"):
        raise RuntimeError("render_clipped_after_fit")
    min_font = max(args.min_font_size, minimum_final_font_size(row["language"]))
    if int(info.get("finalFontSize") or 0) < min_font:
        raise RuntimeError("final_font_too_small")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    element = driver.find_element(By.ID, "stage")
    element.screenshot(str(raw_path))
    quality = evaluate_render_quality(raw_path, bg, info, row["language"])
    info.update(quality)
    if not quality["quality_pass"]:
        raise RuntimeError("render_quality_failed:" + ",".join(quality["quality_reasons"]))
    info.update(crop_rendered_line(raw_path, info, args))
    return info


def write_review(output_dir: Path, metadata: list[dict], sample_count: int, rng: random.Random) -> None:
    sample = list(metadata)
    rng.shuffle(sample)
    sample = sample[:sample_count]
    cards = []
    for row in sample:
        text = html.escape(row["logical_text"])
        cards.append(
            "<section class='card'>"
            f"<img src='{row['raw_image']}' alt='{row['id']}'>"
            f"<div><b>{row['language'].upper()}</b> {html.escape(row['id'])} | {html.escape(row['background_type'])} | {html.escape(row['font_name'])}</div>"
            f"<pre>{text}</pre>"
            "</section>"
        )
    page = """<!doctype html>
<html><head><meta charset="utf-8"><title>Synthetic Raw Review</title>
<style>
body{margin:0;background:#f5f3ee;font-family:Arial,sans-serif;color:#20242a}
header{position:sticky;top:0;background:#fffdf7;border-bottom:1px solid #ddd5c7;padding:12px 18px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px;padding:16px}
.card{background:white;border:1px solid #d8d8d0;border-radius:6px;padding:10px}
img{width:100%;max-height:180px;object-fit:contain;background:#e9e5dc;border:1px solid #ded9cc}
pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0 0;font-size:16px}
</style></head><body>
<header><h1>Synthetic Raw Review</h1><div>Browser raw screenshots before degradation.</div></header>
<main class="grid">
"""
    page += "\n".join(cards)
    page += "\n</main></body></html>\n"
    (output_dir / "review_raw.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists; pass --overwrite to replace it: {args.output_dir}")
        if "synthetic" not in args.output_dir.name:
            raise SystemExit(f"Refusing to overwrite unexpected directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template_path = args.output_dir / "_renderer_template.html"
    write_template(template_path)
    template_url = template_path.resolve().as_uri()

    pools = {language: read_text_pool(args.text_pool_dir, language, args) for language in args.languages}
    text_plan_seed = args.seed if args.text_plan_seed is None else args.text_plan_seed
    text_shard_size = args.text_shard_size or args.per_lang
    pools = apply_text_sharding(
        pools,
        args.text_shard_index,
        args.text_shard_count,
        mode=args.text_shard_mode,
        shard_size=text_shard_size,
        plan_seed=text_plan_seed,
        max_occurrences=args.max_text_occurrences,
    )
    fonts = read_fonts(args.font_manifest)
    backgrounds = read_backgrounds(args.background_manifest)
    by_type: dict[str, list[dict]] = {}
    for bg in backgrounds:
        by_type.setdefault(bg["background_type"], []).append(bg)
    if not backgrounds:
        raise SystemExit("No backgrounds found.")
    metadata = []
    failures = []
    per_lang_counter = Counter()
    attempted_per_lang = Counter()
    text_queues = shuffled_text_queues(pools, args.per_lang, rng, args.languages)
    driver = build_driver(args)
    try:
        while any(per_lang_counter[language] < args.per_lang for language in args.languages):
            needed_languages = [language for language in args.languages if per_lang_counter[language] < args.per_lang]
            language = rng.choice(needed_languages)
            if not text_queues[language]:
                raise SystemExit(
                    f"Ran out of {language} text rows before reaching target. "
                    f"target={args.per_lang}, generated={per_lang_counter[language]}, attempted={attempted_per_lang[language]}"
                )
            row = text_queues[language].pop()
            attempted_per_lang[language] += 1
            sample_id = make_sample_id(language, args.start_index + per_lang_counter[language])
            sample_seed = rng.randrange(1, 2**31 - 1)
            rel_raw = Path("synthetic_raw") / language / f"{sample_id}.png"
            raw_path = args.output_dir / rel_raw
            render_info = None
            bg = None
            font = None
            style = None
            last_error = ""
            for attempt in range(1, args.render_retries + 1):
                local_rng = random.Random(sample_seed + attempt * 1000003)
                bg = choose_background(backgrounds, by_type, row["text"], language, local_rng)
                font = choose_font(fonts, language, local_rng, args.style_profile)
                style = choose_style(row, font, bg, local_rng, args.style_profile)
                try:
                    render_info = render_one(driver, template_url, row, font, bg, style, raw_path, args)
                    break
                except Exception as exc:
                    last_error = str(exc)
            if render_info is None or bg is None or font is None or style is None:
                failures.append(
                    {
                        "id": sample_id,
                        "language": language,
                        "error": last_error or "render_failed",
                        "attempts": args.render_retries,
                        "text": row["text"],
                        "attempted_per_lang": attempted_per_lang[language],
                    }
                )
                continue
            ctc_text_u2, u2_status = (
                ug_visual_label(row["text"], args.require_python_bidi)
                if language == "ug"
                else (row["text"], "ltr_same_as_logical")
            )
            ctc_text = ctc_text_u2 if language == "ug" else row["text"]
            meta = {
                "id": sample_id,
                "shard_id": args.shard_id,
                "text_shard_index": args.text_shard_index,
                "text_shard_count": args.text_shard_count,
                "text_shard_mode": args.text_shard_mode,
                "text_plan_seed": text_plan_seed,
                "text_plan_position": row.get("_text_plan_position"),
                "text_plan_pool_index": row.get("_text_plan_pool_index"),
                "text_plan_occurrence": row.get("_text_plan_occurrence"),
                "text_plan_pool_size": row.get("_text_plan_pool_size"),
                "raw_image": rel_raw.as_posix(),
                "language": language,
                "direction": "rtl" if language == "ug" else "ltr",
                "logical_text": row["text"],
                "ctc_text": ctc_text,
                "ctc_text_u1": row["text"],
                "ctc_text_u2": ctc_text_u2,
                "ctc_text_u2_status": u2_status,
                "text_length": row["text_length"],
                "source": row.get("source", ""),
                "source_group": row.get("source_group", ""),
                "domain": row.get("domain", ""),
                "source_pool_id": row.get("source_pool_id", ""),
                "font": style["font_file"],
                "font_name": style["font_name"],
                "font_path": style["font_path"],
                "background_id": bg["id"],
                "background": bg["image"],
                "background_type": bg["background_type"],
                "background_width": bg["width"],
                "background_height": bg["height"],
                "style_profile": args.style_profile,
                "meme_style": style["meme_style"],
                "contrast_bucket": style["contrast_bucket"],
                "font_size_requested": style["font_size"],
                "font_size_final": render_info.get("finalFontSize"),
                "font_weight": style["font_weight"],
                "font_style": style["font_style"],
                "fill_color": style["fill_color"],
                "stroke_color": style["stroke_color"],
                "stroke_width": style["stroke_width"],
                "shadow": style["shadow_enabled"],
                "shadow_css": style["shadow"],
                "glow": style["glow"],
                "double_effect": style["double_effect"],
                "opacity": style["opacity"],
                "rotation": style["rotation"],
                "skew": style["skew"],
                "letter_spacing": style["letter_spacing"],
                "position_x": style["position_x"],
                "position_y": style["position_y"],
                "backplate_color": style["backplate_color"],
                "backplate_padding_x": style["backplate_padding_x"],
                "backplate_padding_y": style["backplate_padding_y"],
                "backplate_radius": style["backplate_radius"],
                "margin_x": style["margin_x"],
                "margin_y": style["margin_y"],
                "raw_width": render_info.get("crop_box", {}).get("width") or render_info.get("stageBox", {}).get("width"),
                "raw_height": render_info.get("crop_box", {}).get("height") or render_info.get("stageBox", {}).get("height"),
                "raw_uncropped_width": render_info.get("raw_uncropped_width") or render_info.get("stageBox", {}).get("width"),
                "raw_uncropped_height": render_info.get("raw_uncropped_height") or render_info.get("stageBox", {}).get("height"),
                "raw_crop_box": render_info.get("crop_box", {}),
                "text_box": render_info.get("textBox", {}),
                "render_clipped": bool(render_info.get("clipped")),
                "quality_changed_ratio": render_info.get("quality_changed_ratio"),
                "quality_strong_ratio": render_info.get("quality_strong_ratio"),
                "quality_changed_mean_delta": render_info.get("quality_changed_mean_delta"),
                "quality_text_region_std": render_info.get("quality_text_region_std"),
                "quality_min_segment_ratio": render_info.get("quality_min_segment_ratio"),
                "quality_min_segment_strong_ratio": render_info.get("quality_min_segment_strong_ratio"),
                "seed": sample_seed,
                "render_attempts": attempt,
                "generator": "06_generate_samples.py",
            }
            metadata.append(meta)
            per_lang_counter[language] += 1
            if len(metadata) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "generated": len(metadata),
                            "per_lang": dict(per_lang_counter),
                            "failures": len(failures),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        driver.quit()

    (args.output_dir / "metadata_raw.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in metadata) + ("\n" if metadata else ""),
        encoding="utf-8",
    )
    (args.output_dir / "render_failures.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in failures) + ("\n" if failures else ""),
        encoding="utf-8",
    )
    labels_dir = args.output_dir / "labels_raw"
    labels_dir.mkdir(parents=True, exist_ok=True)
    all_lines = []
    for language in args.languages:
        lang_lines = [f"{row['raw_image']}\t{row['ctc_text']}" for row in metadata if row["language"] == language]
        (labels_dir / f"raw_{language}.txt").write_text("\n".join(lang_lines) + ("\n" if lang_lines else ""), encoding="utf-8")
        all_lines.extend(lang_lines)
    (labels_dir / "raw_all.txt").write_text("\n".join(all_lines) + ("\n" if all_lines else ""), encoding="utf-8")
    write_review(args.output_dir, metadata, args.review_samples, rng)
    summary = {
        "output_dir": str(args.output_dir),
        "samples": len(metadata),
        "failures": len(failures),
        "per_lang": dict(Counter(row["language"] for row in metadata)),
        "target_per_lang": args.per_lang,
        "attempted_per_lang": dict(attempted_per_lang),
        "style_profile": args.style_profile,
        "background_type_counts": dict(Counter(row["background_type"] for row in metadata)),
        "meme_style_counts": dict(Counter(row["meme_style"] for row in metadata)),
        "contrast_counts": dict(Counter(row["contrast_bucket"] for row in metadata)),
        "effect_counts": {
            "stroke": sum(1 for row in metadata if float(row.get("stroke_width", 0)) > 0),
            "shadow": sum(1 for row in metadata if row.get("shadow")),
            "glow": sum(1 for row in metadata if row.get("glow")),
            "double_effect": sum(1 for row in metadata if row.get("double_effect")),
            "translucent": sum(1 for row in metadata if float(row.get("opacity", 1.0)) < 0.999),
            "rotated": sum(1 for row in metadata if abs(float(row.get("rotation", 0))) > 0.001),
            "skewed": sum(1 for row in metadata if abs(float(row.get("skew", 0))) > 0.001),
            "caption_box": sum(1 for row in metadata if row.get("backplate_color") != "transparent"),
        },
        "render_clipped": sum(1 for row in metadata if row.get("render_clipped")),
        "seed": args.seed,
        "text_shard_index": args.text_shard_index,
        "text_shard_count": args.text_shard_count,
        "text_shard_mode": args.text_shard_mode,
        "text_shard_size": text_shard_size,
        "text_plan_seed": text_plan_seed,
        "max_text_occurrences": args.max_text_occurrences,
        "text_candidates_per_lang": {language: len(pools[language]) for language in args.languages},
        "outputs": {
            "metadata_raw": str(args.output_dir / "metadata_raw.jsonl"),
            "review_raw": str(args.output_dir / "review_raw.html"),
            "labels_raw_dir": str(labels_dir),
        },
    }
    (args.output_dir / "summary_raw.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
