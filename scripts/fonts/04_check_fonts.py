#!/usr/bin/env python3
"""Final font audit and browser test page generation for synthesis fonts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont


FONT_SUFFIXES = {".ttf", ".otf", ".ttc", ".otc"}

UG_CORE_CHARS = "ئ ا ە ب پ ت ج چ خ د ر ز ژ س ش غ ف ق ك گ ڭ ل م ن ھ و ۇ ۆ ۈ ۋ ې ى ي".split()
KK_SPECIFIC_CHARS = list("ӘәҒғҚқҢңӨөҰұҮүҺһІі")

UG_TEST_TEXT = "ئۇيغۇرچە سىناق: ئاتا-ئانا، كۆڭۈل، ھېچ ئىش يوق 123 ABC"
KK_TEST_TEXT = "Қазақша сынақ: Әә Ғғ Ққ Ңң Өө Ұұ Үү Һһ Іі 123 ABC"
ZH_TEST_TEXT = "中文字体测试：真实模因行图像，黑体宋体描边阴影 123 ABC"


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--font-root", type=Path, default=defaults["font_root"])
    parser.add_argument("--metadata", type=Path, default=defaults["metadata"])
    parser.add_argument("--text-pool-dir", type=Path, default=defaults["text_pool_dir"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--chrome-binary", type=Path, default=None)
    parser.add_argument("--chromedriver", type=Path, default=None)
    parser.add_argument("--skip-browser", action="store_true")
    return parser.parse_args()


def default_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    return {
        "font_root": svtr_root / "03_synthetic_generation" / "font_library" / "raw_fonts",
        "metadata": svtr_root / "01_data_preparation" / "real_line_dataset_eval_reviewed" / "metadata.csv",
        "text_pool_dir": svtr_root / "02_corpus_preparation" / "mixed_text_pool_v2",
        "output_dir": svtr_root / "03_synthetic_generation" / "font_library" / "final_check",
    }


def load_download_manifest(font_library_root: Path) -> dict[str, dict[str, str]]:
    manifest_path = font_library_root / "download_manifest.json"
    if not manifest_path.exists():
        return {}
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_path = {}
    for item in obj.get("fonts", []):
        by_path[str(Path(item.get("path", "")).resolve()).lower()] = {
            "source_url": item.get("url", ""),
            "license": infer_license(item.get("url", ""), item.get("filename", "")),
            "archive_member": item.get("archive_member", ""),
            "kind": item.get("kind", ""),
        }
    return by_path


def infer_license(url: str, filename: str) -> str:
    low = f"{url} {filename}".lower()
    if "dejavu" in low:
        return "DejaVu Fonts License"
    if "liberation" in low:
        return "SIL Open Font License 1.1"
    if "source-han" in low or "noto" in low or "google/fonts" in low or "amiri" in low:
        return "SIL Open Font License 1.1"
    return "unknown"


def load_required_chars(metadata: Path, text_pool_dir: Path) -> dict[str, set[str]]:
    required: dict[str, set[str]] = defaultdict(set)
    if metadata.exists():
        with metadata.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                language = row.get("language", "")
                text = row.get("text") or row.get("logical_text") or ""
                add_text(required, language, text)
    for language in ("zh", "ug", "kk"):
        pool_path = text_pool_dir / f"{language}_mixed_text_pool_v2.jsonl"
        if not pool_path.exists():
            continue
        with pool_path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                add_text(required, language, obj.get("normalized_text") or obj.get("text") or "")

    required["ug"].update(UG_CORE_CHARS)
    required["kk"].update(KK_SPECIFIC_CHARS)
    return required


def add_text(required: dict[str, set[str]], language: str, text: str) -> None:
    for ch in text:
        if ch.isspace():
            continue
        if language == "zh" and is_han(ch):
            required[language].add(ch)
        elif language == "ug" and is_ug_script(ch):
            required[language].add(ch)
        elif language == "kk" and is_cyrillic(ch):
            required[language].add(ch)


def is_han(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def is_ug_script(ch: str) -> bool:
    return ("\u0600" <= ch <= "\u06ff") or ("\u0750" <= ch <= "\u077f")


def is_cyrillic(ch: str) -> bool:
    return "\u0400" <= ch <= "\u04ff"


def scan_fonts(font_root: Path) -> list[tuple[str, Path]]:
    items = []
    for language in ("zh", "ug", "kk"):
        lang_dir = font_root / language
        if not lang_dir.exists():
            continue
        for path in sorted(lang_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in FONT_SUFFIXES:
                items.append((language, path))
    return items


def iter_faces(path: Path):
    if path.suffix.lower() in {".ttc", ".otc"}:
        collection = TTCollection(str(path), lazy=True)
        for idx, font in enumerate(collection.fonts):
            yield idx, font
    else:
        yield 0, TTFont(str(path), lazy=True)


def font_name(font: TTFont, fallback: str) -> str:
    if "name" not in font:
        return fallback
    for name_id in (16, 1, 4):
        for record in font["name"].names:
            if record.nameID != name_id:
                continue
            try:
                value = record.toUnicode().strip()
            except Exception:
                continue
            if value:
                return value
    return fallback


def cmap_chars(font: TTFont) -> set[str]:
    values = set()
    if "cmap" not in font:
        return values
    for table in font["cmap"].tables:
        values.update(chr(cp) for cp in table.cmap.keys() if 0 <= cp <= sys.maxunicode)
    return values


def coverage(required: set[str], available: set[str]) -> tuple[float, list[str]]:
    missing = sorted(required - available, key=lambda ch: ord(ch))
    ratio = (len(required) - len(missing)) / len(required) if required else 1.0
    return ratio, missing


def format_missing(missing: list[str]) -> str:
    return " ".join(f"{ch}(U+{ord(ch):04X})" for ch in missing[:200])


def audit_fonts(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    required = load_required_chars(args.metadata, args.text_pool_dir)
    source_meta = load_download_manifest(args.font_root.parent)
    rows: list[dict[str, object]] = []
    manifest: dict[str, list[dict[str, object]]] = {"zh": [], "ug": [], "kk": []}

    for language, path in scan_fonts(args.font_root):
        for face_index, font in iter_faces(path):
            try:
                name = font_name(font, path.stem)
                available = cmap_chars(font)
            finally:
                font.close()
            cov, missing = coverage(required[language], available)
            passed = cov == 1.0
            path_key = str(path.resolve()).lower()
            meta = source_meta.get(path_key, {})
            row = {
                "font": path.name,
                "font_name": name,
                "language": language,
                "face_index": face_index,
                "coverage": cov,
                "missing_characters": missing,
                "missing_count": len(missing),
                "pass": passed,
                "font_path": str(path),
                "license": meta.get("license", "unknown"),
                "source_url": meta.get("source_url", ""),
                "archive_member": meta.get("archive_member", ""),
            }
            rows.append(row)
            if passed:
                manifest[language].append({
                    "font": path.name,
                    "font_name": name,
                    "language": language,
                    "font_path": str(path),
                    "face_index": face_index,
                    "license": row["license"],
                    "source_url": row["source_url"],
                })
    return rows, manifest


def write_reports(rows: list[dict[str, object]], manifest: dict[str, list[dict[str, object]]], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "font_check_results.jsonl"
    csv_path = args.output_dir / "font_check_results.csv"
    manifest_path = args.output_dir / "font_manifest_final.json"
    summary_path = args.output_dir / "font_check_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            out = dict(row)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "font", "font_name", "language", "face_index", "coverage",
            "missing_count", "pass", "missing_characters_preview",
            "font_path", "license", "source_url", "archive_member",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "font": row["font"],
                "font_name": row["font_name"],
                "language": row["language"],
                "face_index": row["face_index"],
                "coverage": f"{float(row['coverage']):.6f}",
                "missing_count": row["missing_count"],
                "pass": "yes" if row["pass"] else "no",
                "missing_characters_preview": format_missing(row["missing_characters"]),
                "font_path": row["font_path"],
                "license": row["license"],
                "source_url": row["source_url"],
                "archive_member": row["archive_member"],
            })

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    summary = {
        "font_root": str(args.font_root),
        "metadata": str(args.metadata),
        "text_pool_dir": str(args.text_pool_dir),
        "outputs": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "manifest": str(manifest_path),
        },
        "required": required_counts(args),
        "counts": {
            "total_faces": len(rows),
            "by_language": dict(Counter(str(row["language"]) for row in rows)),
            "passed_by_language": {language: len(items) for language, items in manifest.items()},
            "failed_by_language": dict(Counter(str(row["language"]) for row in rows if not row["pass"])),
        },
        "browser_test": {},
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def required_counts(args: argparse.Namespace) -> dict[str, int]:
    required = load_required_chars(args.metadata, args.text_pool_dir)
    return {language: len(chars) for language, chars in required.items()}


def relative_url(path: Path, html_dir: Path) -> str:
    rel = path.resolve().relative_to(html_dir.resolve()) if path.resolve().is_relative_to(html_dir.resolve()) else path.resolve()
    return Path(rel).as_posix()


def make_browser_html(manifest: dict[str, list[dict[str, object]]], args: argparse.Namespace) -> Path:
    html_dir = args.output_dir / "browser_test"
    html_dir.mkdir(parents=True, exist_ok=True)
    html_path = html_dir / "font_browser_test.html"
    rows = []
    css_blocks = []
    sample_text = {"zh": ZH_TEST_TEXT, "ug": UG_TEST_TEXT, "kk": KK_TEST_TEXT}
    dir_map = {"zh": "ltr", "ug": "rtl", "kk": "ltr"}
    lang_attr = {"zh": "zh", "ug": "ug", "kk": "kk"}

    for language, fonts in manifest.items():
        for idx, font in enumerate(fonts):
            family = f"ocr_{language}_{idx:03d}"
            font_path = Path(str(font["font_path"]))
            css_blocks.append(
                "@font-face { "
                f"font-family: '{family}'; "
                f"src: url('{font_path.resolve().as_uri()}') format('{font_format(font_path)}'); "
                "font-display: block; "
                "}"
            )
            rows.append(
                f"<section class='font-card' data-language='{language}' data-family='{family}'>"
                f"<div class='meta'><b>{html.escape(language.upper())}</b> "
                f"{html.escape(str(font['font_name']))} "
                f"<span>{html.escape(font_path.name)}</span></div>"
                f"<div class='sample' lang='{lang_attr[language]}' dir='{dir_map[language]}' "
                f"style=\"font-family:'{family}';\">{html.escape(sample_text[language])}</div>"
                "<div class='note'></div>"
                "</section>"
            )

    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Font Browser Test</title>
<style>
{chr(10).join(css_blocks)}
body {{ margin: 0; padding: 24px; background: #f6f5f0; color: #111; font-family: Arial, sans-serif; }}
h1 {{ margin: 0 0 16px; font-size: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 14px; }}
.font-card {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
.meta {{ font-size: 13px; color: #333; margin-bottom: 8px; }}
.meta span {{ color: #666; margin-left: 8px; }}
.sample {{
  min-height: 76px;
  padding: 12px 16px;
  font-size: 42px;
  line-height: 1.35;
  background: #171717;
  color: #fff;
  text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000, 0 2px 4px rgba(0,0,0,.6);
  overflow-wrap: anywhere;
}}
.note {{ font-size: 12px; color: #555; margin-top: 6px; }}
</style>
</head>
<body>
<h1>Font Browser Test</h1>
<div class="grid">
{chr(10).join(rows)}
</div>
<script>
document.fonts.ready.then(() => {{
  for (const card of document.querySelectorAll('.font-card')) {{
    const family = card.dataset.family;
    const sample = card.querySelector('.sample').textContent;
    const ok = document.fonts.check('42px ' + family, sample);
    card.querySelector('.note').textContent = 'document.fonts.check=' + ok + '; computed=' + getComputedStyle(card.querySelector('.sample')).fontFamily;
  }}
}});
</script>
</body>
</html>
"""
    html_path.write_text(page, encoding="utf-8")
    return html_path


def font_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".otf":
        return "opentype"
    if suffix == ".ttf":
        return "truetype"
    return "truetype"


def try_browser_screenshot(html_path: Path, args: argparse.Namespace) -> dict[str, object]:
    if args.skip_browser:
        return {"status": "skipped"}
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
    except Exception as exc:
        return {"status": "selenium_unavailable", "error": str(exc), "html": str(html_path)}

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1800,2600")
    if args.chrome_binary:
        options.binary_location = str(args.chrome_binary)
    service = Service(str(args.chromedriver)) if args.chromedriver else None
    screenshot_path = html_path.parent / "font_browser_test.png"
    try:
        driver = webdriver.Chrome(service=service, options=options)
        try:
            driver.get(html_path.resolve().as_uri())
            driver.execute_script("return document.fonts.ready")
            height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)")
            driver.set_window_size(1800, min(max(int(height) + 100, 900), 12000))
            driver.save_screenshot(str(screenshot_path))
        finally:
            driver.quit()
        return {"status": "ok", "html": str(html_path), "screenshot": str(screenshot_path)}
    except Exception as exc:
        return {"status": "browser_failed", "error": str(exc), "html": str(html_path)}


def update_summary_browser(args: argparse.Namespace, browser_result: dict[str, object]) -> None:
    summary_path = args.output_dir / "font_check_summary.json"
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    obj["browser_test"] = browser_result
    summary_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, manifest = audit_fonts(args)
    write_reports(rows, manifest, args)
    html_path = make_browser_html(manifest, args)
    browser_result = try_browser_screenshot(html_path, args)
    update_summary_browser(args, browser_result)

    summary = json.loads((args.output_dir / "font_check_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
