import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_train_auto_from_render_boxes import write_train_review_pages


def default_dataset_root():
    svtr_root = Path(__file__).resolve().parents[2]
    return svtr_root / "01_data_preparation" / "real_line_dataset"


def read_metadata(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {exc}") from exc
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild paginated train_review annotation HTML from existing train_auto metadata.jsonl."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=3000)
    args = parser.parse_args()

    metadata_path = args.metadata or (args.dataset_root / "metadata.jsonl")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found: {metadata_path}")

    rows = read_metadata(metadata_path)
    manifest = write_train_review_pages(rows, args.dataset_root, args.chunk_size)
    summary = {
        "dataset_root": str(args.dataset_root),
        "metadata": str(metadata_path),
        "chunk_size": args.chunk_size,
        "metadata_rows": len(rows),
        "train_review": manifest,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
