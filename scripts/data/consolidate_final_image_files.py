import argparse
import json
import os
import re
import shutil
from pathlib import Path


JSON_NAMES = [
    "train_ug.json", "train_kk.json",
    "dev_ug.json", "dev_kk.json",
    "test_ug.json", "test_kk.json",
]
MARKERS = (
    "Misogyny_Dataset_Project/",
    "Meme_Dataset_Project/",
    "final_multilingual_meme_ocr_dataset/",
)


def safe_part(value):
    return re.sub(r"[^0-9A-Za-z_.()\- \u4e00-\u9fff]+", "_", str(value)).strip(" _") or "unknown"


def rel_from_marker(path_text):
    normalized = str(path_text).replace("\\", "/")
    for marker in MARKERS:
        if marker in normalized:
            return normalized[normalized.index(marker):]
    return None


def resolve_path(value, workspace_root):
    if not value:
        return None
    raw = str(value).replace("/home/data_home/wudayu", "/data_home/wudayu")
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    candidate = workspace_root / raw
    if candidate.exists():
        return candidate
    rel = rel_from_marker(raw)
    if rel:
        candidate = workspace_root / rel
        if candidate.exists():
            return candidate
        for meme_prefix in (
            "Meme_Dataset_Project/images/train images/",
            "Meme_Dataset_Project/images/test images/",
        ):
            if rel.startswith(meme_prefix):
                candidate = workspace_root / "Meme_Dataset_Project" / "images" / "meme" / rel[len(meme_prefix):]
                if candidate.exists():
                    return candidate
    return None


def choose_path(record, keys, workspace_root):
    for key in keys:
        p = resolve_path(record.get(key), workspace_root)
        if p is not None:
            return p
    return None


def destination_for(src, kind, dataset_dir):
    rel = rel_from_marker(src)
    if rel is None:
        rel = safe_part(src.name)
    # Keep enough original structure to avoid filename collisions.
    rel_path = Path(*[safe_part(part) for part in Path(rel).parts])
    return dataset_dir / "full_images" / kind / rel_path


def link_or_copy(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        if mode == "hardlink":
            raise
        shutil.copy2(src, dst)
        return "copy_fallback"


def process_record(record, dataset_dir, workspace_root, mode, dry_run):
    outputs = []
    for kind, keys, abs_key, workspace_key, simple_key in [
        ("source", ["source_image_abs", "source_image_workspace", "source_image"], "source_image_abs", "source_image_workspace", "source_image"),
        ("rendered", ["rendered_image_abs", "rendered_image_workspace", "rendered_image"], "rendered_image_abs", "rendered_image_workspace", "rendered_image"),
    ]:
        src = choose_path(record, keys, workspace_root)
        if src is None:
            outputs.append({"kind": kind, "status": "missing"})
            continue
        dst = destination_for(src, kind, dataset_dir)
        rel_to_workspace = dst.relative_to(workspace_root).as_posix() if workspace_root in dst.parents else dst.as_posix()
        status = "dry_run"
        if not dry_run:
            status = link_or_copy(src, dst, mode)
        record[f"{kind}_image_consolidated_abs"] = str(dst)
        record[f"{kind}_image_consolidated_workspace"] = rel_to_workspace
        if kind == "source":
            record[abs_key] = str(dst)
            record[workspace_key] = rel_to_workspace
            record[simple_key] = rel_to_workspace
        else:
            record[abs_key] = str(dst)
            record[workspace_key] = rel_to_workspace
            record[simple_key] = rel_to_workspace
        outputs.append({"kind": kind, "status": status, "src": str(src), "dst": str(dst)})
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Consolidate full source/rendered images referenced by final JSONs into final_multilingual_meme_ocr_dataset/full_images.")
    parser.add_argument("--workspace-root", type=Path, default=Path("D:/text_renderer"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("D:/text_renderer/final_multilingual_meme_ocr_dataset"))
    parser.add_argument("--mode", choices=["hardlink", "copy", "copy-fallback"], default="copy-fallback")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-names", nargs="*", default=JSON_NAMES)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    workspace_root = args.workspace_root.resolve()
    out_json_dir = dataset_dir / "json_consolidated"
    manifest_path = dataset_dir / "full_images_manifest.jsonl"
    summary = {"records": 0, "image_events": 0, "missing": 0, "by_status": {}, "outputs": []}
    mode = "copy" if args.mode == "copy" else "copy-fallback"
    if args.mode == "hardlink":
        mode = "hardlink"

    if not args.dry_run:
        out_json_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_lines = []
    for name in args.json_names:
        src_json = dataset_dir / name
        if not src_json.exists():
            continue
        data = json.loads(src_json.read_text(encoding="utf-8"))
        for record in data:
            summary["records"] += 1
            events = process_record(record, dataset_dir, workspace_root, mode, args.dry_run)
            for event in events:
                summary["image_events"] += 1
                status = event.get("status", "unknown")
                summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
                if status == "missing":
                    summary["missing"] += 1
                event["json"] = name
                event["sample_id"] = record.get("sample_id", "")
                manifest_lines.append(json.dumps(event, ensure_ascii=False))
        out_path = out_json_dir / name
        summary["outputs"].append(str(out_path))
        if not args.dry_run:
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = dataset_dir / "full_images_consolidation_summary.json"
    if not args.dry_run:
        manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


