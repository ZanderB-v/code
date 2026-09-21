import argparse
import csv
from pathlib import Path

from build_pilot_annotation_html import build_html, read_rows

INVALID = '<>:"/\\|?*'


def safe_part(name):
    return ''.join('_' if ch in INVALID else ch for ch in name)


def safe_rel(path_value):
    if not path_value:
        return path_value
    parts = str(path_value).replace('\\', '/').split('/')
    return '/'.join(safe_part(part) for part in parts)


def rename_files(root):
    renamed = []
    for base_name in ['crops', 'overlays']:
        base = root / base_name
        if not base.exists():
            continue
        files = sorted([p for p in base.rglob('*') if p.is_file()], key=lambda p: len(p.parts), reverse=True)
        for path in files:
            safe_name = safe_part(path.name)
            if safe_name == path.name:
                continue
            target = path.with_name(safe_name)
            if target.exists():
                path.unlink()
                renamed.append((str(path), str(target), 'removed_duplicate'))
            else:
                path.rename(target)
                renamed.append((str(path), str(target), 'renamed'))
    return renamed


def rewrite_csv(path):
    if not path.exists():
        return 0
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    changed = 0
    for row in rows:
        for field in ['crop_path_rel', 'overlay_path_rel']:
            old = row.get(field, '')
            new = safe_rel(old)
            if new != old:
                row[field] = new
                changed += 1
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return changed


def rebuild_html(root):
    ok_csv = root / 'candidates_ok.csv'
    html_path = root / 'annotate.html'
    rows = read_rows(ok_csv)
    html_path.write_text(build_html(rows), encoding='utf-8')
    return len(rows)


def repair_subset(root):
    renamed = rename_files(root)
    csv_changes = {}
    for name in ['selected_candidates_before_crop.csv', 'candidates.csv', 'candidates_ok.csv']:
        csv_changes[name] = rewrite_csv(root / name)
    html_rows = rebuild_html(root) if (root / 'candidates_ok.csv').exists() else 0
    return {
        'subset': str(root),
        'renamed_files': len(renamed),
        'csv_changes': csv_changes,
        'html_rows': html_rows,
    }


def main():
    parser = argparse.ArgumentParser(description='Rename crop/overlay files to Windows-safe names and rebuild annotate.html.')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--subsets', nargs='+', default=['dev_reviewed', 'test_reviewed'])
    args = parser.parse_args()
    for subset in args.subsets:
        print(repair_subset(args.dataset_root / subset))


if __name__ == '__main__':
    main()
