import argparse
import json
import random
from pathlib import Path

from generate_html_css_selenium_smoke import (
    CounterLike,
    build_driver,
    render_one,
    write_review,
)


def read_pool(path, language):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = (obj.get("text") or obj.get("target") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "language": language,
                    "source": obj.get("source") or "synthetic_text_pool",
                    "source_sample_id": obj.get("id") or obj.get("source_row_id") or "",
                    "line_index": obj.get("phrase_index") or "",
                    "domain": obj.get("domain") or "",
                    "text_length": obj.get("text_length") or len(text),
                }
            )
    return rows


def sample_targets(pools, per_lang, rng):
    targets = []
    for language, rows in pools.items():
        if len(rows) < per_lang:
            raise SystemExit(f"Not enough {language} pool rows: need {per_lang}, got {len(rows)}")
        targets.extend(rng.sample(rows, per_lang))
    rng.shuffle(targets)
    return targets


def default_paths():
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    corpus_root = svtr_root / "02_corpus_preparation" / "synthetic_text_pool"
    out_dir = svtr_root / "03_synthetic_generation" / "smoke_pool_30"
    return {
        "pool_dir": corpus_root,
        "output_dir": out_dir,
    }


def main():
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="Generate HTML/CSS/Selenium smoke images from zh/ug/kk text pools.")
    parser.add_argument("--pool-dir", type=Path, default=defaults["pool_dir"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--per-lang", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--chrome-binary", default="")
    parser.add_argument("--driver-path", default="")
    parser.add_argument("--no-sandbox", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pools = {
        "zh": read_pool(args.pool_dir / "zh_text_pool_50000.jsonl", "zh"),
        "ug": read_pool(args.pool_dir / "ug_text_pool_50000.jsonl", "ug"),
        "kk": read_pool(args.pool_dir / "kk_text_pool_50000.jsonl", "kk"),
    }
    targets = sample_targets(pools, args.per_lang, rng)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    label_lines = []
    driver = build_driver(args)
    try:
        for i, row in enumerate(targets, 1):
            language = row["language"]
            sample_id = f"{language}_{i:06d}"
            rel_image = Path("images") / language / f"{sample_id}.png"
            out_path = args.output_dir / rel_image
            out_path.parent.mkdir(parents=True, exist_ok=True)
            style = render_one(driver, row, out_path, rng)
            meta = {
                "id": sample_id,
                "image": str(rel_image).replace("\\", "/"),
                "language": language,
                "logical_text": row["text"],
                "ctc_text": row["text"],
                "source": row["source"],
                "source_sample_id": row["source_sample_id"],
                "line_index": row["line_index"],
                "domain": row.get("domain", ""),
                "text_length": row.get("text_length", len(row["text"])),
                "renderer": "html_css_selenium_pool_smoke_v1",
                "seed": args.seed,
                **style,
            }
            metadata.append(meta)
            label_lines.append(f"{meta['image']}\t{meta['ctc_text']}")
    finally:
        driver.quit()

    (args.output_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in metadata) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "labels.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "count": len(metadata),
                "output_dir": str(args.output_dir),
                "languages": CounterLike(metadata),
                "per_lang": args.per_lang,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_review(args.output_dir, metadata)
    print(json.dumps({"count": len(metadata), "output_dir": str(args.output_dir), "review_html": str(args.output_dir / "review.html")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
