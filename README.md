# Multilingual Meme Line Recognition (SOAR-SVTR)

Code-only snapshot of the multilingual meme text line-recognition project.
The main method extends SVTRv2 with visual-order CTC and logical-order semantic
guidance, cross-order consistency, and script-conditioned residual adaptation.

This repository contains experiment scripts, model implementation, configuration
files, documentation, the 4,891-character recognition dictionary, and the
third-party source trees required by the baseline comparison. It intentionally
does **not** contain cropped images, raw datasets, LMDBs, model checkpoints,
pretrained weights, training logs, evaluation predictions, or Test data.

Most experiment YAMLs and scripts retain the original server's absolute paths.
Set paths for your own environment before running; the snapshot alone cannot
reproduce results without the separately held data and weights. See
`00_docs/SOAR_CODE_LEVEL_NOVELTY_AUDIT.md` for implementation ownership and
`00_docs/MULTISCRIPT_DUAL_ORDER_SGM_V1.md` for the method protocol.

The vendored OpenOCR, MMOCR, and PARSeq code remains third-party material.
Consult their respective `LICENSE` files and source-provenance metadata before
redistributing or modifying it. Scientific results and dataset access are not
part of this code publication.
