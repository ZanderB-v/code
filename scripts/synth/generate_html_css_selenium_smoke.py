import argparse
import html
import json
import random
import re
import unicodedata
from pathlib import Path
from urllib.parse import quote

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


LANG_TARGET_DEFAULT = {"zh": 4, "ug": 3, "kk": 3}

FONT_STACKS = {
    "zh": "Microsoft YaHei, SimHei, Noto Sans CJK SC, Arial, sans-serif",
    "ug": "Microsoft Uighur, UKIJ Tuz, Noto Naskh Arabic, Noto Sans Arabic, Arial, sans-serif",
    "kk": "Arial, Times New Roman, Noto Sans, DejaVu Sans, sans-serif",
}

FILL_COLORS = ["#ffffff", "#fff5d6", "#111111", "#ffea00", "#ff3b30", "#00e5ff", "#f5f5f5"]
STROKE_COLORS = ["#000000", "#ffffff", "#5a1a00", "#1c1c1c", "#003a66"]
BG_STYLES = [
    "linear-gradient(120deg, #2b2d42, #8d99ae)",
    "linear-gradient(120deg, #f8f9fa, #adb5bd)",
    "linear-gradient(120deg, #4f000b, #ffba08)",
    "linear-gradient(120deg, #001219, #0a9396)",
    "radial-gradient(circle at 30% 40%, #ffffff, #b8c0ff 45%, #22223b)",
    "repeating-linear-gradient(135deg, #f2f2f2 0 12px, #dedede 12px 24px)",
]


def normalize_text(text):
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def record_split(record):
    return record.get("final_split") or record.get("split") or ""


def collect_texts(index_path, max_per_lang=2000):
    records = json.loads(Path(index_path).read_text(encoding="utf-8"))
    by_lang = {"zh": [], "ug": [], "kk": []}
    seen = {"zh": set(), "ug": set(), "kk": set()}
    for record in records:
        if record_split(record) != "train":
            continue
        for line in record.get("texts") or []:
            values = {
                "zh": line.get("zh_text") or line.get("source_text") or "",
                "ug": line.get("render_text_normalized") or line.get("ug_text") or line.get("text") or "",
                "kk": line.get("kk_text") or "",
            }
            for lang, text in values.items():
                text = normalize_text(text)
                if not text or text in seen[lang]:
                    continue
                length = len(text.replace(" ", ""))
                if lang == "zh" and not (3 <= length <= 35):
                    continue
                if lang in ("ug", "kk") and not (3 <= length <= 45):
                    continue
                seen[lang].add(text)
                by_lang[lang].append({
                    "text": text,
                    "language": lang,
                    "source": "final_train_index",
                    "source_sample_id": record.get("sample_id") or "",
                    "line_index": line.get("line_index", ""),
                })
                if all(len(by_lang[x]) >= max_per_lang for x in by_lang):
                    return by_lang
    return by_lang


def font_size_for(text, lang, rng):
    length = len(text.replace(" ", ""))
    base = 68 if length <= 10 else 58 if length <= 20 else 48 if length <= 32 else 42
    if lang == "zh":
        base += 2
    return max(34, base + rng.randint(-5, 5))


def make_style(text, lang, rng):
    fill = rng.choice(FILL_COLORS)
    stroke = rng.choice(STROKE_COLORS)
    bg = rng.choice(BG_STYLES)
    font_size = font_size_for(text, lang, rng)
    stroke_width = rng.choice([0, 1, 2, 3])
    shadow = rng.choice([
        "none",
        "2px 2px 0 rgba(0,0,0,.65)",
        "0 0 8px rgba(255,255,255,.9)",
        "3px 3px 6px rgba(0,0,0,.55)",
    ])
    rotation = rng.uniform(-2.0, 2.0)
    opacity = rng.choice([1.0, 1.0, 1.0, 0.92])
    return {
        "fill": fill,
        "stroke": stroke,
        "background": bg,
        "font_size": font_size,
        "stroke_width": stroke_width,
        "shadow": shadow,
        "rotation": rotation,
        "opacity": opacity,
        "font_family": FONT_STACKS[lang],
    }


def make_html(text, lang, style):
    direction = "rtl" if lang == "ug" else "ltr"
    align = "right" if lang == "ug" else "left"
    escaped = html.escape(text)
    return f"""<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  body {{ width: max-content; height: max-content; overflow: hidden; }}
  #line {{
    display: inline-block;
    max-width: 1100px;
    box-sizing: border-box;
    padding: 18px 28px 20px 28px;
    background: {style['background']};
    color: {style['fill']};
    font-family: {style['font_family']};
    font-size: {style['font_size']}px;
    line-height: 1.18;
    font-weight: 700;
    white-space: nowrap;
    direction: {direction};
    unicode-bidi: plaintext;
    text-align: {align};
    -webkit-text-stroke: {style['stroke_width']}px {style['stroke']};
    text-shadow: {style['shadow']};
    opacity: {style['opacity']};
    transform: rotate({style['rotation']:.3f}deg);
    transform-origin: center center;
  }}
</style>
</head>
<body>
  <div id="line" lang="{lang}" dir="{direction}">{escaped}</div>
</body>
</html>"""


def build_driver(args):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--hide-scrollbars")
    options.add_argument("--force-device-scale-factor=1")
    options.add_argument("--window-size=1400,260")
    options.add_argument("--allow-file-access-from-files")
    options.add_argument("--disable-dev-shm-usage")
    if args.no_sandbox:
        options.add_argument("--no-sandbox")
    if args.chrome_binary:
        options.binary_location = args.chrome_binary
    service = Service(args.driver_path) if args.driver_path else Service()
    return webdriver.Chrome(service=service, options=options)


def render_one(driver, row, out_path, rng):
    style = make_style(row["text"], row["language"], rng)
    page = make_html(row["text"], row["language"], style)
    driver.get("data:text/html;charset=utf-8," + quote(page))
    driver.execute_async_script("const done = arguments[0]; document.fonts.ready.then(() => done());")
    elem = driver.find_element(By.ID, "line")
    elem.screenshot(str(out_path))
    return style


def write_review(output_dir, metadata):
    rows = []
    for item in metadata:
        rel = item["image"].replace("\\", "/")
        direction = "rtl" if item["language"] == "ug" else "ltr"
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['id'])}</td>"
            f"<td>{html.escape(item['language'])}</td>"
            f"<td dir='{direction}'>{html.escape(item['logical_text'])}</td>"
            f"<td><img src='{html.escape(rel)}'></td>"
            "</tr>"
        )
    content = """<!doctype html><html><head><meta charset='utf-8'><title>Synthetic line smoke</title>
<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;vertical-align:top}img{max-width:720px;max-height:130px;background:#eee}</style>
</head><body><h1>HTML/CSS/Selenium Synthetic Line Smoke</h1><table><thead><tr><th>ID</th><th>Lang</th><th>Label</th><th>Image</th></tr></thead><tbody>""" + "".join(rows) + """</tbody></table></body></html>"""
    (output_dir / "review.html").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate a 10-image HTML/CSS/Selenium synthetic line smoke set from existing dataset text.")
    parser.add_argument("--index", type=Path, default=Path("/data_home/wudayu/final_multilingual_meme_ocr_dataset/train_ug.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition/01_data_preparation/synthetic_smoke/html_css_10"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--chrome-binary", default="")
    parser.add_argument("--driver-path", default="")
    parser.add_argument("--no-sandbox", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = args.output_dir
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    by_lang = collect_texts(args.index)
    targets = []
    remaining = args.count
    for lang, n in LANG_TARGET_DEFAULT.items():
        take = min(n, remaining)
        if take <= 0:
            break
        if len(by_lang[lang]) < take:
            raise SystemExit(f"Not enough {lang} texts: need {take}, got {len(by_lang[lang])}")
        targets.extend(rng.sample(by_lang[lang], take))
        remaining -= take
    while remaining > 0:
        lang = rng.choice(["zh", "ug", "kk"])
        targets.append(rng.choice(by_lang[lang]))
        remaining -= 1
    rng.shuffle(targets)

    metadata = []
    label_lines = []
    driver = build_driver(args)
    try:
        for i, row in enumerate(targets, 1):
            lang = row["language"]
            sample_id = f"{lang}_{i:06d}"
            rel_image = Path("images") / lang / f"{sample_id}.png"
            out_path = output_dir / rel_image
            out_path.parent.mkdir(parents=True, exist_ok=True)
            style = render_one(driver, row, out_path, rng)
            meta = {
                "id": sample_id,
                "image": str(rel_image).replace("\\", "/"),
                "language": lang,
                "logical_text": row["text"],
                "ctc_text": row["text"],
                "source": row["source"],
                "source_sample_id": row["source_sample_id"],
                "line_index": row["line_index"],
                "renderer": "html_css_selenium_smoke_v1",
                "seed": args.seed,
                **style,
            }
            metadata.append(meta)
            label_lines.append(f"{meta['image']}\t{meta['ctc_text']}")
    finally:
        driver.quit()

    (output_dir / "metadata.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in metadata) + "\n", encoding="utf-8")
    (output_dir / "labels.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps({"count": len(metadata), "output_dir": str(output_dir), "languages": CounterLike(metadata)}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(output_dir, metadata)
    print(json.dumps({"count": len(metadata), "output_dir": str(output_dir), "review_html": str(output_dir / "review.html")}, ensure_ascii=False, indent=2))


def CounterLike(metadata):
    counts = {}
    for item in metadata:
        counts[item["language"]] = counts.get(item["language"], 0) + 1
    return counts


if __name__ == "__main__":
    main()
