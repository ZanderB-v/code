#!/usr/bin/env python3
"""Build a native Visio architecture diagram matching the supplied reference.

The output is intentionally made from separate Visio shapes instead of a
single raster image, so labels, arrows, panels, and feature-map blocks remain
editable in Microsoft Visio.
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


VISIO_NS = "http://schemas.microsoft.com/office/visio/2012/main"
ET.register_namespace("", VISIO_NS)

PAGE_W = 16.6666666667
PAGE_H = 11.6666666667

COLORS = {
    "ink": "#172536",
    "muted": "#4F5E6D",
    "blue_fill": "#EFF6FD",
    "blue": "#4D78B2",
    "blue_dark": "#214D9A",
    "green_fill": "#F0F9EF",
    "green": "#5B9360",
    "green_dark": "#357443",
    "orange_fill": "#FFF5DE",
    "orange": "#E59A25",
    "orange_dark": "#B35C10",
    "purple_fill": "#F8F0FF",
    "purple": "#8060AF",
    "purple_dark": "#68458F",
    "gray_fill": "#F7F8FA",
    "gray": "#89939D",
    "white": "#FFFFFF",
}


def vsdx_classes():
    try:
        from vsdx import VisioFile  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: python -m pip install vsdx") from exc
    return VisioFile


def template_path() -> Path:
    import vsdx  # type: ignore

    return Path(vsdx.__file__).resolve().parent / "media" / "media.vsdx"


def set_text_style(shape, size=0.14, color=COLORS["ink"], bold=False, font="Noto Sans SC"):
    char = shape.xml.find(f"{{{VISIO_NS}}}Section[@N='Character']")
    if char is None:
        char = ET.SubElement(shape.xml, f"{{{VISIO_NS}}}Section", {"N": "Character"})
    row = char.find(f"{{{VISIO_NS}}}Row")
    if row is None:
        row = ET.SubElement(char, f"{{{VISIO_NS}}}Row", {"IX": "0"})
    for name, value in {
        "Font": font,
        "Size": str(size),
        "Color": color,
        "Style": "1" if bold else "0",
    }.items():
        cell = row.find(f"{{{VISIO_NS}}}Cell[@N='{name}']")
        if cell is None:
            cell = ET.SubElement(row, f"{{{VISIO_NS}}}Cell", {"N": name})
        cell.set("V", value)

    para = shape.xml.find(f"{{{VISIO_NS}}}Section[@N='Paragraph']")
    if para is None:
        para = ET.SubElement(shape.xml, f"{{{VISIO_NS}}}Section", {"N": "Paragraph"})
    prow = para.find(f"{{{VISIO_NS}}}Row")
    if prow is None:
        prow = ET.SubElement(para, f"{{{VISIO_NS}}}Row", {"IX": "0"})
    align = prow.find(f"{{{VISIO_NS}}}Cell[@N='HorzAlign']")
    if align is None:
        align = ET.SubElement(prow, f"{{{VISIO_NS}}}Cell", {"N": "HorzAlign"})
    align.set("V", "1")


def style_shape(shape, fill, line, *, size=0.14, bold=False, dashed=False, text_color=None):
    shape.fill_color = fill
    shape.line_color = line
    shape.line_weight = 0.012
    if dashed:
        shape.set_cell_value("LinePattern", "2")
    set_text_style(shape, size=size, color=text_color or COLORS["ink"], bold=bold)


def build_vsdx(output: Path):
    VisioFile = vsdx_classes()
    output.parent.mkdir(parents=True, exist_ok=True)
    template = template_path()
    backup = output.with_name(output.stem + "_original_backup.vsdx")
    if output.exists() and not backup.exists():
        shutil.copy2(output, backup)
    shutil.copyfile(template, output)

    with VisioFile(str(output)) as vis:
        page = vis.pages[0]
        page.name = "SVTRv2-Based Multilingual OCR"
        page.width = PAGE_W
        page.height = PAGE_H
        for shape in list(page.child_shapes):
            shape.remove()
        stale_connects = page.xml.find(f"{{{VISIO_NS}}}Connects")
        if stale_connects is not None:
            page.xml.getroot().remove(stale_connects)

        media = VisioFile(str(template))
        media_page = media.pages[0]
        rect_template = next(s for s in media_page.child_shapes if s.text.strip() == "RECTANGLE")
        line_template = next(s for s in media_page.child_shapes if s.text.strip() == "LINE")
        circle_template = next(s for s in media_page.child_shapes if s.text.strip() == "CIRCLE")

        def box(x, y, w, h, text="", fill=COLORS["white"], line=COLORS["gray"], *, size=0.14, bold=False, dashed=False, text_color=None):
            shape = rect_template.copy(page)
            shape.x, shape.y, shape.width, shape.height = x, y, w, h
            shape.text = text
            style_shape(shape, fill, line, size=size, bold=bold, dashed=dashed, text_color=text_color)
            return shape

        def circle(x, y, d, text, fill, line, *, size=0.14, bold=True):
            shape = circle_template.copy(page)
            shape.x, shape.y, shape.width, shape.height = x, y, d, d
            shape.text = text
            style_shape(shape, fill, line, size=size, bold=bold)
            return shape

        def arrow(x1, y1, x2, y2, color=COLORS["muted"], *, dashed=False, weight=0.018):
            shape = line_template.copy(page)
            shape.text = ""
            shape.set_start_and_finish((x1, y1), (x2, y2))
            shape.line_color = color
            shape.line_weight = weight
            shape.end_arrow = True
            if dashed:
                shape.set_cell_value("LinePattern", "2")
            return shape

        def tile(x, y, w=0.12, h=0.18, fill=COLORS["blue_fill"], line=COLORS["blue"]):
            return box(x, y, w, h, "", fill, line, size=0.04)

        # Header.
        box(8.33, 11.15, 15.7, 0.38,
            "Overview of the Proposed Multilingual Meme Text Recognition Model (SVTRv2-Based)",
            COLORS["white"], COLORS["white"], size=0.235, bold=True)
        box(8.33, 10.76, 8.0, 0.28,
            "for Chinese, Uyghur, and Kazakh Line-Text OCR",
            COLORS["white"], COLORS["white"], size=0.17, text_color=COLORS["ink"])

        # Left input panel.
        box(1.28, 7.35, 2.25, 4.45, "", COLORS["white"], COLORS["ink"], size=0.12)
        box(1.28, 9.55, 1.85, 0.34, "Input multilingual\nmeme text line", COLORS["white"], COLORS["white"], size=0.125, bold=True)
        box(1.28, 8.87, 1.72, 0.22, "Chinese", COLORS["white"], COLORS["white"], size=0.105, bold=True)
        box(1.28, 8.45, 1.92, 0.56, "生活不止眼前的苟且", COLORS["gray_fill"], COLORS["gray"], size=0.105)
        box(1.28, 7.78, 1.82, 0.22, "Uyghur (RTL)", COLORS["white"], COLORS["white"], size=0.105, bold=True)
        box(1.28, 7.36, 1.92, 0.56, "ئۇيغۇرچە سۆزلەر", COLORS["gray_fill"], COLORS["gray"], size=0.11)
        box(1.28, 6.69, 1.92, 0.22, "Kazakh (Cyrillic)", COLORS["white"], COLORS["white"], size=0.105, bold=True)
        box(1.28, 6.27, 1.92, 0.56, "Армансыз адам - қанатсыз құс.", COLORS["gray_fill"], COLORS["gray"], size=0.085)

        # MSR module.
        box(3.35, 7.35, 1.40, 4.10, "", COLORS["blue_fill"], COLORS["blue"], size=0.12)
        box(3.35, 8.28, 1.14, 1.55, "MSR:\nMulti-Size\nResizing", COLORS["blue_fill"], COLORS["blue_fill"], size=0.14, bold=True)
        # Nested dashed resize candidates.
        box(3.38, 6.55, 0.75, 0.55, "", COLORS["white"], COLORS["blue"], size=0.08, dashed=True)
        box(3.55, 6.67, 0.90, 0.76, "", COLORS["white"], COLORS["blue"], size=0.08, dashed=True)
        box(3.74, 6.83, 1.08, 1.02, "", COLORS["white"], COLORS["blue"], size=0.08, dashed=True)
        box(3.35, 5.70, 1.15, 0.24, "dynamic width", COLORS["white"], COLORS["white"], size=0.09, text_color=COLORS["muted"])

        # SVTRv2 visual encoder.
        box(5.75, 7.33, 2.82, 5.25, "", COLORS["blue_fill"], COLORS["blue"], size=0.12)
        box(5.75, 9.72, 2.3, 0.32, "SVTRv2 Visual Encoder", COLORS["blue_fill"], COLORS["blue_fill"], size=0.14, bold=True)
        box(5.75, 9.15, 2.18, 0.72, "Patch Embedding", COLORS["white"], COLORS["blue"], size=0.125, bold=True)
        for i in range(6):
            tile(4.93 + i * 0.25, 9.12, fill=COLORS["blue_fill"], line=COLORS["blue"])
        arrow(5.75, 8.77, 5.75, 8.58, color=COLORS["blue_dark"])
        for y, label in [(8.25, "Stage 1\n(local / global mixing)"), (7.48, "Stage 2\n(local / global mixing)"), (6.71, "Stage 3\n(local / global mixing)")]:
            box(5.75, y, 2.15, 0.55, label, COLORS["white"], COLORS["blue"], size=0.105, bold=True)
        arrow(5.75, 7.95, 5.75, 7.77, color=COLORS["blue_dark"])
        arrow(5.75, 7.18, 5.75, 7.00, color=COLORS["blue_dark"])
        box(5.75, 5.85, 2.10, 0.60, "FRM: Feature\nRearrangement Module", COLORS["orange_fill"], COLORS["orange"], size=0.105, bold=True)
        box(5.75, 5.36, 2.10, 0.22, "(local / global mixing)", COLORS["blue_fill"], COLORS["blue_fill"], size=0.09, text_color=COLORS["muted"])
        arrow(5.75, 6.42, 5.75, 6.16, color=COLORS["blue_dark"])
        # Feature map between encoder and script module.
        for ix in range(3):
            for iy in range(4):
                tile(7.52 + ix * 0.13, 6.10 + iy * 0.13, w=0.12, h=0.12, fill=COLORS["blue_fill"], line=COLORS["blue"])

        # Local script adaptation.
        box(8.86, 7.36, 2.65, 4.05, "", COLORS["green_fill"], COLORS["green"], size=0.12)
        box(8.86, 9.35, 2.15, 0.46, "Script-Conditioned\nAdaptation", COLORS["green_fill"], COLORS["green_fill"], size=0.135, bold=True)
        box(8.86, 8.86, 2.10, 0.24, "(script-aware adapters)", COLORS["green_fill"], COLORS["green_fill"], size=0.09, text_color=COLORS["muted"])
        rows = [(8.25, "汉", "Han", COLORS["green_fill"], COLORS["green"]),
                (7.57, "ع", "Arabic", COLORS["purple_fill"], COLORS["purple"]),
                (6.89, "С", "Cyrillic", COLORS["blue_fill"], COLORS["blue"]),
                (6.21, "A", "Latin", COLORS["orange_fill"], COLORS["orange"])]
        for y, glyph, label, fill, line in rows:
            box(8.13, y, 0.28, 0.32, glyph, fill, line, size=0.14, bold=True)
            box(8.62, y, 0.65, 0.22, label, COLORS["green_fill"], COLORS["green_fill"], size=0.10, bold=True)
            for ix in range(3):
                box(9.38 + ix * 0.18, y, 0.10, 0.34, "", fill, line, size=0.05)
        circle(10.67, 6.65, 0.34, "+", COLORS["white"], COLORS["ink"], size=0.18)
        for ix in range(3):
            for iy in range(3):
                tile(11.06 + ix * 0.13, 6.49 + iy * 0.13, w=0.12, h=0.12, fill=COLORS["green_fill"], line=COLORS["green"])
        box(10.78, 5.90, 0.82, 0.28, "Adapted\nFeature Map", COLORS["green_fill"], COLORS["green_fill"], size=0.085, bold=True)
        arrow(7.95, 6.43, 8.00, 6.43, color=COLORS["blue_dark"])

        # Branching to CTC and SGM.
        arrow(11.48, 6.65, 11.82, 8.06, color=COLORS["ink"])
        arrow(11.48, 6.35, 11.82, 4.98, color=COLORS["ink"])

        # CTC branch and final prediction.
        box(12.30, 8.20, 2.45, 2.22, "", COLORS["blue_fill"], COLORS["blue"], size=0.12)
        box(12.30, 9.20, 2.08, 0.56, "CTC Branch\n(main inference branch)", COLORS["blue_fill"], COLORS["blue_fill"], size=0.13, bold=True)
        box(12.30, 8.62, 1.75, 0.22, "U2 visual order", COLORS["blue_fill"], COLORS["blue_fill"], size=0.10, bold=True, text_color=COLORS["blue_dark"])
        tile(11.35, 8.00, fill=COLORS["blue_fill"], line=COLORS["blue"])
        tile(11.50, 8.00, fill=COLORS["blue_fill"], line=COLORS["blue"])
        tile(11.35, 7.82, fill=COLORS["blue_fill"], line=COLORS["blue"])
        box(13.38, 7.78, 1.18, 0.62, "Linear Classifier\n+ RCTC Loss", COLORS["white"], COLORS["blue"], size=0.095, bold=True)
        arrow(12.05, 7.92, 12.84, 7.92, color=COLORS["blue_dark"])

        box(15.08, 8.20, 1.88, 2.25, "", COLORS["green_fill"], COLORS["green"], size=0.12)
        box(15.08, 9.37, 1.55, 0.28, "Final Prediction", COLORS["green_fill"], COLORS["green_fill"], size=0.13, bold=True)
        box(15.08, 8.72, 1.58, 0.46, "生 活 不 止 眼 前 ... 苟 且", COLORS["white"], COLORS["gray"], size=0.075)
        box(15.08, 8.36, 1.15, 0.20, "(visual order)", COLORS["green_fill"], COLORS["green_fill"], size=0.085, text_color=COLORS["muted"])
        arrow(14.76, 8.20, 15.08, 8.20, color=COLORS["ink"])
        arrow(15.08, 7.68, 15.08, 7.33, color=COLORS["blue_dark"])

        # Cross-order consistency.
        box(12.30, 6.43, 2.45, 0.92, "Cross-Order Consistency\nbidi permutation alignment\n+ JS consistency loss", COLORS["orange_fill"], COLORS["orange"], size=0.105, bold=True)
        arrow(12.30, 7.08, 12.30, 6.90, color=COLORS["blue_dark"])

        # SGM semantic guidance (training only).
        box(12.28, 4.33, 2.52, 2.70, "", COLORS["purple_fill"], COLORS["purple"], size=0.12, dashed=True)
        box(12.28, 5.67, 2.15, 0.50, "SGM: Semantic Guidance Module", COLORS["purple_fill"], COLORS["purple_fill"], size=0.105, bold=True)
        box(12.28, 5.27, 1.42, 0.20, "(train only)", COLORS["purple_fill"], COLORS["purple_fill"], size=0.09, text_color=COLORS["muted"])
        box(12.28, 4.95, 1.58, 0.22, "Unicode logical order", COLORS["purple_fill"], COLORS["purple_fill"], size=0.095, bold=True, text_color=COLORS["purple_dark"])
        tile(11.42, 4.35, fill=COLORS["purple_fill"], line=COLORS["purple"])
        tile(11.57, 4.35, fill=COLORS["purple_fill"], line=COLORS["purple"])
        tile(11.42, 4.17, fill=COLORS["purple_fill"], line=COLORS["purple"])
        box(13.40, 4.47, 1.13, 0.70, "Decoder + LM\n+ CE Loss", COLORS["purple_fill"], COLORS["purple"], size=0.095, bold=True)
        arrow(12.08, 4.50, 12.86, 4.50, color=COLORS["purple_dark"])
        box(12.28, 3.82, 1.84, 0.30, "生 活 不 止 眼 前 ... 苟 且", COLORS["white"], COLORS["gray"], size=0.075)
        box(12.28, 3.50, 1.25, 0.18, "(logical order)", COLORS["purple_fill"], COLORS["purple_fill"], size=0.08, text_color=COLORS["muted"])
        box(12.28, 3.16, 1.00, 0.18, "(train only)", COLORS["purple_fill"], COLORS["purple_fill"], size=0.08, text_color=COLORS["purple_dark"])
        arrow(12.30, 4.98, 12.30, 5.12, color=COLORS["purple_dark"])
        arrow(13.02, 5.95, 13.02, 6.43, color=COLORS["purple_dark"])

        # Optional local direction conditioning.
        box(15.00, 4.45, 1.38, 1.95, "Optional\nLocal Direction\nConditioning\n\nnegative ablation,\nnot used in final model", COLORS["gray_fill"], COLORS["gray"], size=0.092, bold=False, dashed=True)
        arrow(14.80, 4.83, 15.00, 4.83, color=COLORS["gray"], dashed=True)
        arrow(15.08, 7.02, 15.08, 6.40, color=COLORS["green_dark"], dashed=True)
        box(15.00, 6.60, 1.38, 0.92, "Inference: SGM removed,\nonly the CTC branch is kept.", COLORS["green_fill"], COLORS["green"], size=0.085, dashed=True)

        # Input and feature-map flow arrows, added last for visibility.
        arrow(2.40, 7.35, 2.62, 7.35, color=COLORS["ink"])
        arrow(4.10, 7.35, 4.34, 7.35, color=COLORS["ink"])
        arrow(7.20, 6.43, 7.95, 6.43, color=COLORS["ink"])

        # Contributions band.
        box(6.80, 1.42, 12.35, 1.75, "", COLORS["white"], "#B5BEC7", size=0.10)
        box(1.18, 2.25, 1.55, 0.22, "Our Key Contributions", COLORS["white"], COLORS["white"], size=0.11, bold=True, text_color=COLORS["blue_dark"])
        cards = [
            (2.45, 1.38, 3.35, COLORS["purple_fill"], COLORS["purple"], "1", "Dual-Order Semantic Guidance", "Use visual order (U2) and logical order\n(Unicode) together for rich semantic supervision."),
            (6.45, 1.38, 3.35, COLORS["orange_fill"], COLORS["orange"], "2", "Cross-Order Consistency", "Align the two orders via bidi permutation\nand JS consistency loss for robust recognition."),
            (10.45, 1.38, 4.25, COLORS["green_fill"], COLORS["green"], "3", "Script-Conditioned Adaptation", "Script-aware adapters specialize features for\nHan, Arabic, Cyrillic, Latin, and mixed runs."),
        ]
        for x, y, w, fill, line, number, title, body in cards:
            box(x, y, w, 1.02, "", fill, line, size=0.09)
            circle(x - w / 2 + 0.34, y + 0.24, 0.32, number, line, line, size=0.15)
            box(x + 0.20, y + 0.29, w - 0.88, 0.22, title, fill, fill, size=0.095, bold=True, text_color=line)
            box(x + 0.20, y - 0.12, w - 0.72, 0.38, body, fill, fill, size=0.075)

        # Legend.
        box(15.05, 1.42, 2.35, 2.12, "", COLORS["white"], COLORS["gray"], size=0.10, dashed=True)
        box(15.05, 2.25, 1.65, 0.22, "Data flow", COLORS["white"], COLORS["white"], size=0.09, bold=True)
        arrow(14.48, 2.25, 14.85, 2.25, color=COLORS["ink"])
        box(15.05, 1.88, 1.65, 0.22, "Train only", COLORS["white"], COLORS["white"], size=0.09, bold=True, text_color=COLORS["purple_dark"])
        arrow(14.48, 1.88, 14.85, 1.88, color=COLORS["purple_dark"], dashed=True)
        box(15.05, 1.50, 1.65, 0.22, "Optional / Ablation", COLORS["white"], COLORS["white"], size=0.09, bold=True, text_color=COLORS["gray"])
        arrow(14.48, 1.50, 14.85, 1.50, color=COLORS["gray"], dashed=True)
        tile(14.46, 1.08, fill=COLORS["blue_fill"], line=COLORS["blue"])
        tile(14.65, 1.08, fill=COLORS["green_fill"], line=COLORS["green"])
        tile(14.84, 1.08, fill=COLORS["purple_fill"], line=COLORS["purple"])
        box(15.35, 1.08, 1.05, 0.18, "Feature maps", COLORS["white"], COLORS["white"], size=0.08)

        vis.save_vsdx(str(output))
        media.close_vsdx()


def draw_preview(output: Path):
    """Draw a lightweight visual preview with the same major geometry."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.font_manager import FontProperties, fontManager  # type: ignore
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # type: ignore

    cjk_font = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
    if cjk_font.exists():
        fontManager.addfont(str(cjk_font))
        plt.rcParams["font.family"] = FontProperties(fname=str(cjk_font)).get_name()

    png = output.with_name(output.stem + "_preview.png")
    fig = plt.figure(figsize=(14, 10), dpi=150, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.axis("off")

    def rect(x, y, w, h, text, fill, edge, size=9, bold=False, dashed=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=fill, edgecolor=edge, linewidth=1.3,
                                    linestyle="--" if dashed else "-"))
        ax.text(x, y, text, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", color=COLORS["ink"], linespacing=1.12)

    def line(x1, y1, x2, y2, color=COLORS["muted"], dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.5, linestyle="--" if dashed else "-", color=color,
                                     shrinkA=3, shrinkB=3))

    ax.text(0.32, 11.24, "Overview of the Proposed Multilingual Meme Text Recognition Model (SVTRv2-Based)", fontsize=18, fontweight="bold", color=COLORS["ink"], ha="left")
    ax.text(0.32, 10.82, "for Chinese, Uyghur, and Kazakh Line-Text OCR", fontsize=12, color=COLORS["ink"], ha="left", style="italic")
    rect(1.28, 7.35, 2.25, 4.45, "Input multilingual\nmeme text line\n\nChinese\n生活不止眼前的苟且\n\nUyghur (RTL)\nئۇيغۇرچە سۆزلەر\n\nKazakh (Cyrillic)\nАрмансыз адам - қанатсыз құс.", COLORS["white"], COLORS["ink"], size=8, bold=False)
    rect(3.35, 7.35, 1.40, 4.10, "MSR:\nMulti-Size\nResizing\n\n\n[ nested resize ]\n\ndynamic width", COLORS["blue_fill"], COLORS["blue"], size=9, bold=True)
    rect(5.75, 7.33, 2.82, 5.25, "SVTRv2 Visual Encoder\n\nPatch Embedding\n\nStage 1\n(local / global mixing)\n\nStage 2\n(local / global mixing)\n\nStage 3\n(local / global mixing)\n\nFRM: Feature\nRearrangement Module", COLORS["blue_fill"], COLORS["blue"], size=8.5, bold=False)
    rect(8.86, 7.36, 2.65, 4.05, "Script-Conditioned\nAdaptation\n(script-aware adapters)\n\n汉  Han  |||\nع  Arabic |||\nС  Cyrillic |||\nA  Latin |||\n\n+  Adapted Feature Map", COLORS["green_fill"], COLORS["green"], size=9, bold=False)
    rect(12.30, 8.20, 2.45, 2.22, "CTC Branch\n(main inference branch)\n\nU2 visual order\n\nLinear Classifier\n+ RCTC Loss", COLORS["blue_fill"], COLORS["blue"], size=9, bold=True)
    rect(15.08, 8.20, 1.88, 2.25, "Final Prediction\n\n生 活 不 止 眼 前 ... 苟 且\n\n(visual order)", COLORS["green_fill"], COLORS["green"], size=8.5, bold=True)
    rect(12.30, 6.43, 2.45, 0.92, "Cross-Order Consistency\nbidi permutation alignment\n+ JS consistency loss", COLORS["orange_fill"], COLORS["orange"], size=8.5, bold=True)
    rect(12.28, 4.33, 2.52, 2.70, "SGM: Semantic Guidance Module\n(train only)\nUnicode logical order\n\nDecoder + LM\n+ CE Loss\n\n生 活 不 止 眼 前 ... 苟 且\n(logical order)", COLORS["purple_fill"], COLORS["purple"], size=8.3, dashed=True)
    rect(15.00, 4.45, 1.38, 1.95, "Optional\nLocal Direction\nConditioning\n\nnegative ablation,\nnot used in final model", COLORS["gray_fill"], COLORS["gray"], size=7.8, dashed=True)
    rect(15.00, 6.60, 1.38, 0.92, "Inference: SGM removed,\nonly the CTC branch is kept.", COLORS["green_fill"], COLORS["green"], size=7.5, dashed=True)
    for a in [(2.40, 7.35, 2.62, 7.35), (4.10, 7.35, 4.34, 7.35), (7.20, 6.43, 7.95, 6.43), (11.48, 6.65, 11.82, 8.06), (11.48, 6.35, 11.82, 4.98), (14.76, 8.20, 15.08, 8.20), (12.30, 7.08, 12.30, 6.90), (13.02, 5.95, 13.02, 6.43)]:
        line(*a, color=COLORS["ink"])
    line(14.80, 4.83, 15.00, 4.83, color=COLORS["gray"], dashed=True)
    rect(6.80, 1.42, 12.35, 1.75, "Our Key Contributions", COLORS["white"], "#B5BEC7", size=10, bold=True)
    rect(2.45, 1.22, 3.35, 0.92, "1  Dual-Order Semantic Guidance\nVisual order (U2) + logical order (Unicode)", COLORS["purple_fill"], COLORS["purple"], size=7.5, bold=True)
    rect(6.45, 1.22, 3.35, 0.92, "2  Cross-Order Consistency\nBidi permutation + JS consistency loss", COLORS["orange_fill"], COLORS["orange"], size=7.5, bold=True)
    rect(10.45, 1.22, 4.25, 0.92, "3  Script-Conditioned Adaptation\nHan, Arabic, Cyrillic, Latin, and mixed runs", COLORS["green_fill"], COLORS["green"], size=7.5, bold=True)
    rect(15.05, 1.42, 2.35, 2.12, "Data flow  ->\nTrain only  - - ->\nOptional / Ablation  - - ->\n\nFeature maps", COLORS["white"], COLORS["gray"], size=8, dashed=True)
    fig.savefig(png, dpi=150, facecolor="white")
    plt.close(fig)
    return png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output .vsdx path")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    build_vsdx(output)
    preview = draw_preview(output)
    print("REFERENCE_STYLE_ARCHITECTURE_VSDX_OK")
    print(f"PREVIEW_BYTES={preview.stat().st_size}")


if __name__ == "__main__":
    main()
