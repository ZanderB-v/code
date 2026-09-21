#!/usr/bin/env python3
"""Download the recommended font candidates for synthetic OCR rendering."""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path


DIRECT_FONTS = [
    # Uyghur, robust Arabic fonts.
    ("ug", "NotoNaskhArabic-wght.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/notonaskharabic/NotoNaskhArabic%5Bwght%5D.ttf"),
    ("ug", "NotoSansArabic-wdth-wght.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansarabic/NotoSansArabic%5Bwdth,wght%5D.ttf"),
    ("ug", "ScheherazadeNew-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/scheherazadenew/ScheherazadeNew-Regular.ttf"),
    ("ug", "ScheherazadeNew-Bold.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/scheherazadenew/ScheherazadeNew-Bold.ttf"),
    ("ug", "Amiri-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/amiri/Amiri-Regular.ttf"),
    ("ug", "Amiri-Bold.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/amiri/Amiri-Bold.ttf"),
    # Uyghur, limited enhanced set.
    ("ug", "NotoKufiArabic-wght.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/notokufiarabic/NotoKufiArabic%5Bwght%5D.ttf"),
    ("ug", "IBMPlexSansArabic-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/ibmplexsansarabic/IBMPlexSansArabic-Regular.ttf"),
    ("ug", "IBMPlexSansArabic-Bold.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/ibmplexsansarabic/IBMPlexSansArabic-Bold.ttf"),
    ("ug", "ElMessiri-wght.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/elmessiri/ElMessiri%5Bwght%5D.ttf"),

    # Kazakh Cyrillic, robust families.
    ("kk", "NotoSans-wdth-wght.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans%5Bwdth,wght%5D.ttf"),
    ("kk", "NotoSerif-wdth-wght.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/notoserif/NotoSerif%5Bwdth,wght%5D.ttf"),
    ("kk", "NotoSansMono-wdth-wght.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansmono/NotoSansMono%5Bwdth,wght%5D.ttf"),
    ("kk", "NotoSerifDisplay-wdth-wght.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/notoserifdisplay/NotoSerifDisplay%5Bwdth,wght%5D.ttf"),
    ("kk", "PTSans-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/ptsans/PT_Sans-Web-Regular.ttf"),
    ("kk", "PTSans-Bold.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/ptsans/PT_Sans-Web-Bold.ttf"),
    ("kk", "PTSerif-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/ptserif/PT_Serif-Web-Regular.ttf"),
    ("kk", "PTSerif-Bold.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/ptserif/PT_Serif-Web-Bold.ttf"),

    # Chinese, robust CJK families.
    ("zh", "NotoSansCJKsc-Regular.otf", "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"),
    ("zh", "NotoSansCJKsc-Bold.otf", "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf"),
    ("zh", "NotoSansCJKsc-Black.otf", "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Black.otf"),
    ("zh", "NotoSerifCJKsc-Regular.otf", "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf"),
    ("zh", "NotoSerifCJKsc-Bold.otf", "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Bold.otf"),
    ("zh", "SourceHanSansSC-Regular.otf", "https://raw.githubusercontent.com/adobe-fonts/source-han-sans/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"),
    ("zh", "SourceHanSansSC-Bold.otf", "https://raw.githubusercontent.com/adobe-fonts/source-han-sans/release/OTF/SimplifiedChinese/SourceHanSansSC-Bold.otf"),
    ("zh", "SourceHanSansSC-Heavy.otf", "https://raw.githubusercontent.com/adobe-fonts/source-han-sans/release/OTF/SimplifiedChinese/SourceHanSansSC-Heavy.otf"),
    ("zh", "SourceHanSerifSC-Regular.otf", "https://raw.githubusercontent.com/adobe-fonts/source-han-serif/release/OTF/SimplifiedChinese/SourceHanSerifSC-Regular.otf"),
    ("zh", "SourceHanSerifSC-Bold.otf", "https://raw.githubusercontent.com/adobe-fonts/source-han-serif/release/OTF/SimplifiedChinese/SourceHanSerifSC-Bold.otf"),
]

ARCHIVES = [
    {
        "name": "dejavu-fonts-ttf-2.37.zip",
        "url": "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip",
        "language": "kk",
        "members": [
            "dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf",
            "dejavu-fonts-ttf-2.37/ttf/DejaVuSans-Bold.ttf",
            "dejavu-fonts-ttf-2.37/ttf/DejaVuSerif.ttf",
            "dejavu-fonts-ttf-2.37/ttf/DejaVuSerif-Bold.ttf",
        ],
    },
    {
        "name": "liberation-fonts-ttf-2.1.5.tar.gz",
        "url": "https://github.com/liberationfonts/liberation-fonts/files/7261482/liberation-fonts-ttf-2.1.5.tar.gz",
        "language": "kk",
        "members": [
            "liberation-fonts-ttf-2.1.5/LiberationSans-Regular.ttf",
            "liberation-fonts-ttf-2.1.5/LiberationSans-Bold.ttf",
            "liberation-fonts-ttf-2.1.5/LiberationSerif-Regular.ttf",
            "liberation-fonts-ttf-2.1.5/LiberationSerif-Bold.ttf",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=defaults["output_root"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def default_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    return {
        "output_root": svtr_root / "03_synthetic_generation" / "font_library",
    }


def download(url: str, path: Path, force: bool = False) -> str:
    if path.exists() and path.stat().st_size > 0 and not force:
        return "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return "downloaded"


def extract_zip_member(archive_path: Path, member: str, target_path: Path, force: bool) -> str:
    if target_path.exists() and target_path.stat().st_size > 0 and not force:
        return "exists"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        with zf.open(member) as src, target_path.open("wb") as dst:
            dst.write(src.read())
    return "extracted"


def extract_tar_member(archive_path: Path, member: str, target_path: Path, force: bool) -> str:
    if target_path.exists() and target_path.stat().st_size > 0 and not force:
        return "exists"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tf:
        src = tf.extractfile(member)
        if src is None:
            raise FileNotFoundError(member)
        with src, target_path.open("wb") as dst:
            dst.write(src.read())
    return "extracted"


def main() -> None:
    args = parse_args()
    raw_root = args.output_root / "raw_fonts"
    archive_root = args.output_root / "archives"
    records = []

    for language, filename, url in DIRECT_FONTS:
        target = raw_root / language / filename
        status = download(url, target, force=args.force)
        print(f"{status}: {target}")
        records.append({
            "language": language,
            "filename": filename,
            "path": str(target),
            "url": url,
            "status": status,
            "kind": "direct_font",
            "bytes": target.stat().st_size,
        })

    for archive in ARCHIVES:
        archive_path = archive_root / archive["name"]
        archive_status = download(archive["url"], archive_path, force=args.force)
        print(f"{archive_status}: {archive_path}")
        for member in archive["members"]:
            filename = Path(member).name
            target = raw_root / archive["language"] / filename
            if archive_path.suffix == ".zip":
                status = extract_zip_member(archive_path, member, target, force=args.force)
            else:
                status = extract_tar_member(archive_path, member, target, force=args.force)
            print(f"{status}: {target}")
            records.append({
                "language": archive["language"],
                "filename": filename,
                "path": str(target),
                "url": archive["url"],
                "archive_member": member,
                "status": status,
                "kind": "archive_font",
                "bytes": target.stat().st_size,
            })

    manifest_path = args.output_root / "download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({
            "output_root": str(args.output_root),
            "raw_fonts": str(raw_root),
            "archives": str(archive_root),
            "fonts": records,
        }, f, ensure_ascii=False, indent=2)
    print(json.dumps({"fonts": len(records), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
