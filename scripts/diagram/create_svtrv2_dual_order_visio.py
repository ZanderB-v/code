#!/usr/bin/env python3
"""Create an editable Visio diagram and a PNG preview for the proposed M3 model.

The VSDX is built from the small shape templates bundled with the ``vsdx``
package. Every box and connector remains a native Visio object.
"""

from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "00_docs" / "model_diagram_v1"
VSDX_PATH = OUT / "SVTRv2_Multiscript_Dual_Order_M3.vsdx"
PNG_PATH = OUT / "SVTRv2_Multiscript_Dual_Order_M3_preview.png"
SVG_PATH = OUT / "SVTRv2_Multiscript_Dual_Order_M3_preview.svg"

VISIO_NS = "http://schemas.microsoft.com/office/visio/2012/main"
ET.register_namespace("", VISIO_NS)

PAGE_W = 16.0
PAGE_H = 9.0

COLORS = {
    "ink": "#203040",
    "muted": "#5B6B7A",
    "line": "#7890A3",
    "official_fill": "#DCEAF7",
    "official_line": "#3A6F9F",
    "proposed_fill": "#FCE7D1",
    "proposed_line": "#B96319",
    "consistency_fill": "#E9E1F7",
    "consistency_line": "#7254A3",
    "protocol_fill": "#E2F1E7",
    "protocol_line": "#34734A",
    "output_fill": "#DDF0E6",
    "output_line": "#2F7650",
    "ablation_fill": "#F1F2F3",
    "ablation_line": "#7C8791",
    "white": "#FFFFFF",
}


def load_vsdx_classes():
    try:
        from vsdx import VisioFile  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The Python package 'vsdx' is required. Install it with: python -m pip install vsdx"
        ) from exc
    return VisioFile


def direct_template_path() -> Path:
    import vsdx  # type: ignore

    return Path(vsdx.__file__).resolve().parent / "media" / "media.vsdx"


def set_text_style(shape, size: float = 0.15, color: str = COLORS["ink"], bold: bool = False):
    """Add explicit font and paragraph sections so Visio renders consistently."""

    char = shape.xml.find(f"{{{VISIO_NS}}}Section[@N='Character']")
    if char is None:
        char = ET.SubElement(shape.xml, f"{{{VISIO_NS}}}Section", {"N": "Character"})
    row = char.find(f"{{{VISIO_NS}}}Row")
    if row is None:
        row = ET.SubElement(char, f"{{{VISIO_NS}}}Row", {"IX": "0"})
    values = {
        "Font": "Arial",
        "Size": str(size),
        "Color": color,
        "Style": "1" if bold else "0",
    }
    for name, value in values.items():
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


def style_shape(shape, fill: str, line: str, text_color: str, size: float, bold: bool, dashed: bool = False):
    shape.fill_color = fill
    shape.line_color = line
    shape.line_weight = 0.012
    if dashed:
        shape.set_cell_value("LinePattern", "2")
    set_text_style(shape, size=size, color=text_color, bold=bold)


def create_vsdx():
    VisioFile = load_vsdx_classes()
    template_path = direct_template_path()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template_path, VSDX_PATH)

    with VisioFile(str(VSDX_PATH)) as vis:
        page = vis.pages[0]
        page.name = "M3 Proposed Model"
        page.width = PAGE_W
        page.height = PAGE_H

        for shape in list(page.child_shapes):
            shape.remove()
        stale_connects = page.xml.find(f"{{{VISIO_NS}}}Connects")
        if stale_connects is not None:
            page.xml.getroot().remove(stale_connects)

        media = VisioFile(str(template_path))
        media_page = media.pages[0]
        rect_template = next(s for s in media_page.child_shapes if s.text.strip() == "RECTANGLE")
        line_template = next(s for s in media_page.child_shapes if s.text.strip() == "LINE")

        def box(x, y, w, h, text, fill, line, *, size=0.15, bold=False, dashed=False, text_color=None):
            shape = rect_template.copy(page)
            shape.x, shape.y, shape.width, shape.height = x, y, w, h
            shape.text = text
            style_shape(
                shape,
                fill,
                line,
                text_color or COLORS["ink"],
                size,
                bold,
                dashed=dashed,
            )
            return shape

        def arrow(x1, y1, x2, y2, *, color=COLORS["line"], dashed=False, weight=0.018):
            shape = line_template.copy(page)
            shape.text = ""
            shape.set_start_and_finish((x1, y1), (x2, y2))
            shape.line_color = color
            shape.line_weight = weight
            shape.end_arrow = True
            if dashed:
                shape.set_cell_value("LinePattern", "2")
            return shape

        # Title and legend.
        box(
            8.0,
            8.62,
            15.1,
            0.38,
            "Multiscript Dual-Order SVTRv2-S for Meme-Line Text Recognition",
            COLORS["white"],
            COLORS["white"],
            size=0.25,
            bold=True,
        )
        box(
            8.0,
            8.25,
            15.1,
            0.24,
            "M3 primary method | P1/MSR_V3 | SVTRv2-S visual encoder | 4,891-class unified dictionary",
            COLORS["white"],
            COLORS["white"],
            size=0.13,
            text_color=COLORS["muted"],
        )
        box(1.35, 7.72, 0.24, 0.18, "", COLORS["official_fill"], COLORS["official_line"], size=0.1)
        box(2.23, 7.72, 0.24, 0.18, "", COLORS["proposed_fill"], COLORS["proposed_line"], size=0.1)
        box(3.24, 7.72, 0.24, 0.18, "", COLORS["protocol_fill"], COLORS["protocol_line"], size=0.1)
        box(4.45, 7.72, 0.24, 0.18, "", COLORS["ablation_fill"], COLORS["ablation_line"], size=0.1, dashed=True)
        box(1.95, 7.72, 0.8, 0.18, "Official", COLORS["white"], COLORS["white"], size=0.10, text_color=COLORS["muted"])
        box(2.84, 7.72, 0.95, 0.18, "Proposed M3", COLORS["white"], COLORS["white"], size=0.10, text_color=COLORS["muted"])
        box(3.88, 7.72, 1.08, 0.18, "Protocol / data", COLORS["white"], COLORS["white"], size=0.10, text_color=COLORS["muted"])
        box(5.10, 7.72, 1.25, 0.18, "Full: negative ablation", COLORS["white"], COLORS["white"], size=0.10, text_color=COLORS["muted"])

        # Main visual path.
        input_box = box(
            1.30,
            6.43,
            1.65,
            1.05,
            "Target meme\nline image",
            COLORS["protocol_fill"],
            COLORS["protocol_line"],
            size=0.17,
            bold=True,
        )
        msr_box = box(
            3.45,
            6.43,
            2.15,
            1.05,
            "P1/MSR_V3\nratio buckets\ndynamic width",
            COLORS["official_fill"],
            COLORS["official_line"],
            size=0.14,
            bold=True,
        )
        encoder_box = box(
            6.20,
            6.43,
            2.55,
            1.22,
            "SVTRv2-S visual encoder\nSVTRv2LNConvTwo33\nlocal + global feature mixing",
            COLORS["official_fill"],
            COLORS["official_line"],
            size=0.14,
            bold=True,
        )
        script_box = box(
            9.45,
            6.43,
            2.55,
            1.22,
            "M3: Local Script Adaptation\nsoft per-column routing\nHan | Arabic | Cyrillic | Latin",
            COLORS["proposed_fill"],
            COLORS["proposed_line"],
            size=0.135,
            bold=True,
        )
        protocol_box = box(
            13.65,
            6.43,
            2.55,
            1.34,
            "Frozen protocol\nS50 synthetic -> target fine-tune\nClean Dev Macro CER selection\nno test during development",
            COLORS["protocol_fill"],
            COLORS["protocol_line"],
            size=0.125,
            bold=True,
        )
        arrow(2.14, 6.43, 2.36, 6.43)
        arrow(4.53, 6.43, 4.92, 6.43)
        arrow(7.48, 6.43, 8.15, 6.43)
        # The protocol box is an annotation, not a data-flow stage.

        # Branches.
        ctc_box = box(
            6.10,
            4.68,
            2.55,
            1.08,
            "RCTC / FRM branch\nmonotonic visual sequence\nCTC labels: y_vis (U2)",
            COLORS["official_fill"],
            COLORS["official_line"],
            size=0.135,
            bold=True,
        )
        sgm_box = box(
            10.15,
            4.68,
            2.65,
            1.08,
            "SGM / SMTR branch\nsemantic sequence modeling\nSGM labels: y_log (Unicode)",
            COLORS["proposed_fill"],
            COLORS["proposed_line"],
            size=0.135,
            bold=True,
        )
        arrow(9.15, 5.82, 7.38, 5.25, color=COLORS["line"])
        arrow(9.75, 5.82, 10.15, 5.25, color=COLORS["proposed_line"])

        # Cross-order consistency between the branches.
        consistency_box = box(
            8.25,
            3.20,
            4.25,
            1.05,
            "M2: Cross-Order Consistency\nexact bidi permutation pi + soft monotonic transport\nL_xorder = JS(Transport_pi(P_ctc), P_sgm), alpha*",
            COLORS["consistency_fill"],
            COLORS["consistency_line"],
            size=0.12,
            bold=True,
        )
        arrow(7.05, 4.12, 8.70, 3.73, color=COLORS["consistency_line"], dashed=True)
        arrow(11.30, 4.12, 10.95, 3.73, color=COLORS["consistency_line"], dashed=True)

        # Optional Full direction branch.
        full_box = box(
            13.75,
            4.68,
            2.65,
            1.08,
            "Full negative ablation\n+ local direction conditioning\nper-column LTR/RTL field\nnot part of primary M3",
            COLORS["ablation_fill"],
            COLORS["ablation_line"],
            size=0.12,
            bold=True,
            dashed=True,
        )
        arrow(11.95, 6.43, 13.75, 5.25, color=COLORS["ablation_line"], dashed=True)

        # Inference route and losses.
        output_box = box(
            8.25,
            1.74,
            4.25,
            0.88,
            "Inference / evaluation\nCTC visual output -> U2 to logical recovery -> ZH / UG / KK text",
            COLORS["output_fill"],
            COLORS["output_line"],
            size=0.135,
            bold=True,
        )
        arrow(7.38, 4.14, 9.12, 2.18, color=COLORS["output_line"])
        arrow(10.15, 3.20, 10.15, 2.18, color=COLORS["consistency_line"], dashed=True)
        loss_box = box(
            3.35,
            1.74,
            4.45,
            0.88,
            "Joint objective\n0.1 L_ctc + 1.0 L_sgm + alpha* L_xorder + 0.10 L_script\n(+ 0.05 L_direction only in Full)",
            COLORS["consistency_fill"],
            COLORS["consistency_line"],
            size=0.115,
            bold=True,
        )
        arrow(8.25, 3.20, 6.25, 2.18, color=COLORS["consistency_line"], dashed=True)

        # Experimental path strip.
        box(2.0, 0.78, 1.25, 0.50, "B1\nofficial", COLORS["official_fill"], COLORS["official_line"], size=0.11, bold=True)
        box(4.25, 0.78, 1.25, 0.50, "M1\ndual-order", COLORS["proposed_fill"], COLORS["proposed_line"], size=0.11, bold=True)
        box(6.50, 0.78, 1.25, 0.50, "M2\n+ consistency", COLORS["consistency_fill"], COLORS["consistency_line"], size=0.11, bold=True)
        box(8.75, 0.78, 1.25, 0.50, "M3\nprimary", COLORS["proposed_fill"], COLORS["proposed_line"], size=0.11, bold=True)
        box(11.00, 0.78, 1.25, 0.50, "Full\nnegative", COLORS["ablation_fill"], COLORS["ablation_line"], size=0.11, bold=True, dashed=True)
        arrow(3.05, 0.78, 3.63, 0.78, color=COLORS["muted"])
        arrow(5.30, 0.78, 5.88, 0.78, color=COLORS["muted"])
        arrow(7.55, 0.78, 8.13, 0.78, color=COLORS["muted"])
        arrow(9.80, 0.78, 10.38, 0.78, color=COLORS["muted"])
        box(
            14.05,
            0.78,
            2.45,
            0.58,
            "Primary claim: M3\nM2 alpha is selected by clean Dev Macro CER",
            COLORS["protocol_fill"],
            COLORS["protocol_line"],
            size=0.10,
            bold=True,
        )

        vis.save_vsdx(str(VSDX_PATH))
        media.close_vsdx()


def draw_preview():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # type: ignore

    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=COLORS["white"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.axis("off")

    def rect(x, y, w, h, text, fill, outline, *, size=10, bold=False, dashed=False):
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor=fill,
            edgecolor=outline,
            linewidth=1.3,
            linestyle="--" if dashed else "-",
        )
        ax.add_patch(patch)
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=size,
            fontweight="bold" if bold else "normal",
            color=COLORS["ink"],
            linespacing=1.12,
        )

    def connector(x1, y1, x2, y2, color=COLORS["line"], dashed=False):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.5,
                linestyle="--" if dashed else "-",
                color=color,
                shrinkA=3,
                shrinkB=3,
            )
        )

    ax.text(0.35, 8.70, "Multiscript Dual-Order SVTRv2-S for Meme-Line Text Recognition", ha="left", va="center", fontsize=22, fontweight="bold", color=COLORS["ink"])
    ax.text(0.35, 8.32, "M3 primary method | P1/MSR_V3 | SVTRv2-S visual encoder | 4,891-class unified dictionary", ha="left", va="center", fontsize=10, color=COLORS["muted"])
    rect(1.30, 7.72, 0.24, 0.18, "", COLORS["official_fill"], COLORS["official_line"], size=1)
    rect(2.28, 7.72, 0.24, 0.18, "", COLORS["proposed_fill"], COLORS["proposed_line"], size=1)
    rect(3.34, 7.72, 0.24, 0.18, "", COLORS["protocol_fill"], COLORS["protocol_line"], size=1)
    rect(4.46, 7.72, 0.24, 0.18, "", COLORS["ablation_fill"], COLORS["ablation_line"], size=1, dashed=True)
    ax.text(1.48, 7.72, "Official", ha="left", va="center", fontsize=7, color=COLORS["muted"])
    ax.text(2.46, 7.72, "Proposed M3", ha="left", va="center", fontsize=7, color=COLORS["muted"])
    ax.text(3.52, 7.72, "Protocol / data", ha="left", va="center", fontsize=7, color=COLORS["muted"])
    ax.text(4.64, 7.72, "Full: negative ablation", ha="left", va="center", fontsize=7, color=COLORS["muted"])
    rect(1.30, 6.43, 1.65, 1.05, "Target meme\nline image", COLORS["protocol_fill"], COLORS["protocol_line"], size=12, bold=True)
    rect(3.45, 6.43, 2.15, 1.05, "P1/MSR_V3\nratio buckets\ndynamic width", COLORS["official_fill"], COLORS["official_line"], size=11, bold=True)
    rect(6.20, 6.43, 2.55, 1.22, "SVTRv2-S visual encoder\nSVTRv2LNConvTwo33\nlocal + global feature mixing", COLORS["official_fill"], COLORS["official_line"], size=11, bold=True)
    rect(9.45, 6.43, 2.55, 1.22, "M3: Local Script Adaptation\nsoft per-column routing\nHan | Arabic | Cyrillic | Latin", COLORS["proposed_fill"], COLORS["proposed_line"], size=10.5, bold=True)
    rect(13.65, 6.43, 2.55, 1.34, "Frozen protocol\nS50 synthetic -> target fine-tune\nClean Dev Macro CER selection\nno test during development", COLORS["protocol_fill"], COLORS["protocol_line"], size=9.5, bold=True)
    connector(2.14, 6.43, 2.36, 6.43); connector(4.53, 6.43, 4.92, 6.43); connector(7.48, 6.43, 8.15, 6.43)
    rect(6.10, 4.68, 2.55, 1.08, "RCTC / FRM branch\nmonotonic visual sequence\nCTC labels: y_vis (U2)", COLORS["official_fill"], COLORS["official_line"], size=10.5, bold=True)
    rect(10.15, 4.68, 2.65, 1.08, "SGM / SMTR branch\nsemantic sequence modeling\nSGM labels: y_log (Unicode)", COLORS["proposed_fill"], COLORS["proposed_line"], size=10.5, bold=True)
    connector(9.15, 5.82, 7.38, 5.25); connector(9.75, 5.82, 10.15, 5.25, COLORS["proposed_line"])
    rect(8.25, 3.20, 4.25, 1.05, "M2: Cross-Order Consistency\nexact bidi permutation pi + soft monotonic transport\nL_xorder = JS(Transport_pi(P_ctc), P_sgm), alpha*", COLORS["consistency_fill"], COLORS["consistency_line"], size=9.0, bold=True)
    connector(7.05, 4.12, 8.70, 3.73, COLORS["consistency_line"], dashed=True); connector(11.30, 4.12, 10.95, 3.73, COLORS["consistency_line"], dashed=True)
    rect(13.75, 4.68, 2.65, 1.08, "Full negative ablation\n+ local direction conditioning\nper-column LTR/RTL field\nnot part of primary M3", COLORS["ablation_fill"], COLORS["ablation_line"], size=9.0, bold=True, dashed=True)
    connector(11.95, 6.43, 13.75, 5.25, COLORS["ablation_line"], dashed=True)
    rect(8.25, 1.74, 4.25, 0.88, "Inference / evaluation\nCTC visual output -> U2 to logical recovery -> ZH / UG / KK text", COLORS["output_fill"], COLORS["output_line"], size=10.0, bold=True)
    connector(7.38, 4.14, 9.12, 2.18, COLORS["output_line"]); connector(10.15, 3.20, 10.15, 2.18, COLORS["consistency_line"], dashed=True)
    rect(3.35, 1.74, 4.45, 0.88, "Joint objective\n0.1 L_ctc + 1.0 L_sgm + alpha* L_xorder + 0.10 L_script\n(+ 0.05 L_direction only in Full)", COLORS["consistency_fill"], COLORS["consistency_line"], size=9.0, bold=True)
    connector(8.25, 3.20, 6.25, 2.18, COLORS["consistency_line"], dashed=True)
    for x, label, fill, outline in [(2.0, "B1\nofficial", COLORS["official_fill"], COLORS["official_line"]), (4.25, "M1\ndual-order", COLORS["proposed_fill"], COLORS["proposed_line"]), (6.50, "M2\n+ consistency", COLORS["consistency_fill"], COLORS["consistency_line"]), (8.75, "M3\nprimary", COLORS["proposed_fill"], COLORS["proposed_line"]), (11.0, "Full\nnegative", COLORS["ablation_fill"], COLORS["ablation_line"])]:
        rect(x, 0.78, 1.25, 0.50, label, fill, outline, size=8.5, bold=True, dashed=label.startswith("Full"))
    connector(3.05, 0.78, 3.63, 0.78, COLORS["muted"]); connector(5.30, 0.78, 5.88, 0.78, COLORS["muted"]); connector(7.55, 0.78, 8.13, 0.78, COLORS["muted"]); connector(9.80, 0.78, 10.38, 0.78, COLORS["muted"])
    rect(14.05, 0.78, 2.45, 0.58, "Primary claim: M3\nM2 alpha is selected by clean Dev Macro CER", COLORS["protocol_fill"], COLORS["protocol_line"], size=8.5, bold=True)

    fig.savefig(PNG_PATH, dpi=150, facecolor=COLORS["white"])
    fig.savefig(SVG_PATH, facecolor=COLORS["white"])
    plt.close(fig)


def main():
    create_vsdx()
    draw_preview()
    print(f"VSDX={VSDX_PATH}")
    print(f"PNG={PNG_PATH}")
    print(f"SVG={SVG_PATH}")


if __name__ == "__main__":
    main()
