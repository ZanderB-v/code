#!/usr/bin/env python3
"""Paired Clean-Dev error migration analysis for frozen B1 and SOAR-SVTR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


PROTOCOL_ID = "stage6_error_migration_v1"
LANGUAGES = ("zh", "ug", "kk")
EXPECTED_EPOCHS = {"b1": 20, "m3": 34}
EXPECTED_ALPHA = 0.15
BOOTSTRAP_SEED = 20260731
KK_SPECIFIC = set("ӘәҒғҚқҢңӨөҰұҮүҺһІі")
VALID_REVIEW_DECISIONS = {
    "genuine_ocr_error",
    "gt_annotation_error",
    "normalization_issue",
    "ambiguous_image",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV fields for empty output: {path}")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def local_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    marker = "svtrv2_line_recognition"
    parts = path.parts
    if marker in parts:
        candidate = root.joinpath(*parts[parts.index(marker) + 1 :])
        if candidate.exists():
            return candidate.resolve()
    return path


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Percentile input is empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def levenshtein_alignment(reference: str, hypothesis: str) -> list[tuple[str, str, str]]:
    """Return one deterministic minimum-edit alignment.

    Ambiguous paths use the frozen priority substitution, deletion, insertion.
    Matches always take priority over an equally scoring edit path.
    """
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "match"
                continue
            candidates = (
                (dp[i - 1][j - 1] + 1, 0, "sub"),
                (dp[i - 1][j] + 1, 1, "del"),
                (dp[i][j - 1] + 1, 2, "ins"),
            )
            cost, _, operation = min(candidates)
            dp[i][j] = cost
            back[i][j] = operation

    result: list[tuple[str, str, str]] = []
    i, j = n, m
    while i or j:
        operation = back[i][j]
        if operation in ("match", "sub"):
            result.append((operation, reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif operation == "del":
            result.append((operation, reference[i - 1], ""))
            i -= 1
        elif operation == "ins":
            result.append((operation, "", hypothesis[j - 1]))
            j -= 1
        else:
            raise RuntimeError(f"Invalid alignment state at ({i}, {j}): {operation}")
    result.reverse()
    return result


def operation_counts(alignment: Iterable[tuple[str, str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for operation, _, _ in alignment:
        if operation != "match":
            counts[operation] += 1
    return counts


def load_predictions(report_dir: Path, model: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    files = {
        "zh": report_dir / "zh/predictions.jsonl",
        "ug": report_dir / "ug_logical/predictions_logical.jsonl",
        "kk": report_dir / "kk/predictions.jsonl",
    }
    records: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for language, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing frozen Clean-Dev predictions for {model}/{language}: {path}. "
                "Stage 6 does not silently change inference settings."
            )
        hashes[str(path)] = sha256(path)
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("status") != "ok" or raw.get("language") != language:
                    raise ValueError(f"Invalid prediction at {path}:{line_number}")
                image = str(raw["image"]).replace("\\", "/")
                if not image.startswith("dev_reviewed/") or "test" in image.lower():
                    raise ValueError(f"Non-Clean-Dev sample reached Stage 6: {image}")
                key = f"{language}:{image}"
                if key in records:
                    raise ValueError(f"Duplicate prediction key for {model}: {key}")
                if language == "ug":
                    gt = raw["gt_logical_text"]
                    prediction = raw["pred_logical_text"]
                else:
                    gt = raw["eval_gt_text"]
                    prediction = raw["eval_pred_text"]
                alignment = levenshtein_alignment(gt, prediction)
                counts = operation_counts(alignment)
                distance = counts["sub"] + counts["del"] + counts["ins"]
                stored_distance = raw.get(
                    "logical_char_edit_distance" if language == "ug" else "char_edit_distance"
                )
                if stored_distance is not None and int(stored_distance) != distance:
                    raise ValueError(
                        f"Stored/recomputed ED mismatch for {model} {key}: "
                        f"{stored_distance} != {distance}"
                    )
                records[key] = {
                    "model": model,
                    "key": key,
                    "sample_id": Path(image).stem,
                    "language": language,
                    "image_path": image,
                    "gt": gt,
                    "prediction": prediction,
                    "gt_len": len(gt),
                    "pred_len": len(prediction),
                    "ed": distance,
                    "exact": distance == 0,
                    "sub": counts["sub"],
                    "del": counts["del"],
                    "ins": counts["ins"],
                    "alignment": alignment,
                }
    return records, hashes


def validate_summary(
    root: Path,
    summary_path: Path,
    model: str,
    expected_epoch: int,
    records: dict[str, dict[str, Any]],
    require_checkpoint: bool,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    if summary.get("best_epoch") != expected_epoch:
        raise ValueError(
            f"{model} must be CER-selected epoch {expected_epoch}, got {summary.get('best_epoch')}"
        )
    if summary.get("preprocess_protocol") != "P1_MSR_V3":
        raise ValueError(f"{model} is not bound to P1_MSR_V3")
    if summary.get("checkpoint_selection") != "clean_target_dev_macro_CER":
        raise ValueError(f"{model} checkpoint selection protocol drift")
    dev = summary.get("dev", {})
    if dev.get("split") != "dev" or dev.get("dataset_role") != "target_domain":
        raise ValueError(f"{model} summary is not target Clean Dev")
    if "not_evaluated" not in str(summary.get("test_policy", "")):
        raise ValueError(f"{model} summary does not prove Test was excluded")

    checkpoint = local_path(root, summary["best_checkpoint"])
    checkpoint_status = "summary_hash_only"
    if checkpoint.is_file():
        actual_hash = sha256(checkpoint)
        if actual_hash != summary["best_checkpoint_sha256"]:
            raise ValueError(f"{model} checkpoint SHA256 mismatch")
        checkpoint_status = "file_hash_verified"
    elif require_checkpoint:
        raise FileNotFoundError(checkpoint)

    config = local_path(root, summary["config"])
    if not config.is_file():
        raise FileNotFoundError(config)
    if sha256(config) != summary["config_sha256"]:
        raise ValueError(f"{model} config SHA256 mismatch")
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
    if model == "m3":
        if config_payload["Global"].get("method_variant") != "m3":
            raise ValueError("Frozen M3 config has the wrong method variant")
        if not math.isclose(
            float(config_payload["Loss"].get("consistency_weight", -1.0)),
            EXPECTED_ALPHA,
            abs_tol=1e-12,
        ):
            raise ValueError("Frozen M3 config is not alpha=0.15")
        decoder = config_payload["Architecture"]["Decoder"]
        if decoder.get("use_script_adapter") is not True:
            raise ValueError("Frozen M3 config does not enable Script Adaptation")
        if decoder.get("use_local_direction") is not False:
            raise ValueError("Frozen M3 config unexpectedly enables Local Direction Conditioning")

    macro_cer = 0.0
    for language in LANGUAGES:
        language_rows = [row for row in records.values() if row["language"] == language]
        numerator = sum(row["ed"] for row in language_rows)
        denominator = sum(row["gt_len"] for row in language_rows)
        cer = numerator / denominator
        expected_cer = float(dev["languages"][language]["cer"])
        if not math.isclose(cer, expected_cer, abs_tol=1e-12):
            raise ValueError(f"{model}/{language} CER reproduction failed: {cer} != {expected_cer}")
        line_accuracy = sum(row["exact"] for row in language_rows) / len(language_rows)
        expected_line_accuracy = float(dev["languages"][language]["line_accuracy"])
        if not math.isclose(line_accuracy, expected_line_accuracy, abs_tol=1e-12):
            raise ValueError(
                f"{model}/{language} Line Accuracy reproduction failed: "
                f"{line_accuracy} != {expected_line_accuracy}"
            )
        macro_cer += cer / len(LANGUAGES)
    if not math.isclose(macro_cer, float(dev["macro_cer"]), abs_tol=1e-12):
        raise ValueError(f"{model} Macro CER reproduction failed")
    return {
        "summary": summary,
        "summary_sha256": sha256(summary_path),
        "checkpoint_status": checkpoint_status,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": summary["best_checkpoint_sha256"],
        "config": str(config),
        "config_sha256": summary["config_sha256"],
    }


def make_paired_rows(
    root: Path,
    b1: dict[str, dict[str, Any]],
    m3: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if set(b1) != set(m3):
        missing_b1 = sorted(set(m3) - set(b1))[:10]
        missing_m3 = sorted(set(b1) - set(m3))[:10]
        raise ValueError(f"Prediction membership mismatch: missing_b1={missing_b1}, missing_m3={missing_m3}")
    target_root = root / "01_data_preparation/real_line_dataset_eval_reviewed"
    paired: list[dict[str, Any]] = []
    for key in sorted(b1, key=lambda value: (value.split(":", 1)[0], value)):
        base, ours = b1[key], m3[key]
        if base["gt"] != ours["gt"]:
            raise ValueError(f"GT mismatch for {key}")
        if base["exact"] and ours["exact"]:
            migration = "CC"
        elif not base["exact"] and ours["exact"]:
            migration = "WC"
        elif base["exact"] and not ours["exact"]:
            migration = "CW"
        else:
            migration = "WW"
        absolute_image = target_root / base["image_path"]
        if not absolute_image.is_file():
            raise FileNotFoundError(absolute_image)
        paired.append(
            {
                "sample_id": base["sample_id"],
                "language": base["language"],
                "image_path": base["image_path"],
                "absolute_image_path": str(absolute_image.resolve()),
                "gt": base["gt"],
                "pred_b1": base["prediction"],
                "pred_m3": ours["prediction"],
                "gt_len": base["gt_len"],
                "ed_b1": base["ed"],
                "ed_m3": ours["ed"],
                "delta_ed_m3_minus_b1": ours["ed"] - base["ed"],
                "exact_b1": int(base["exact"]),
                "exact_m3": int(ours["exact"]),
                "sub_b1": base["sub"],
                "del_b1": base["del"],
                "ins_b1": base["ins"],
                "sub_m3": ours["sub"],
                "del_m3": ours["del"],
                "ins_m3": ours["ins"],
                "migration": migration,
                "_b1_alignment": base["alignment"],
                "_m3_alignment": ours["alignment"],
            }
        )
    return paired


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def migration_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for language in (*LANGUAGES, "overall"):
        subset = rows if language == "overall" else [row for row in rows if row["language"] == language]
        counts = Counter(row["migration"] for row in subset)
        output.append(
            {
                "language": language,
                "samples": len(subset),
                "stable_correct_CC": counts["CC"],
                "recovered_WC": counts["WC"],
                "regression_CW": counts["CW"],
                "remaining_error_WW": counts["WW"],
                "net_gain_WC_minus_CW": counts["WC"] - counts["CW"],
                "b1_line_accuracy": (counts["CC"] + counts["CW"]) / len(subset),
                "m3_line_accuracy": (counts["CC"] + counts["WC"]) / len(subset),
            }
        )
    return output


def ed_bucket(distance: int) -> str:
    if distance <= 2:
        return f"ED={distance}"
    return "ED>=3"


def edit_distance_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for language in (*LANGUAGES, "overall"):
        subset = rows if language == "overall" else [row for row in rows if row["language"] == language]
        for model, field in (("B1", "ed_b1"), ("M3", "ed_m3")):
            counts = Counter(ed_bucket(int(row[field])) for row in subset)
            for bucket in ("ED=0", "ED=1", "ED=2", "ED>=3"):
                output.append(
                    {
                        "language": language,
                        "model": model,
                        "bucket": bucket,
                        "count": counts[bucket],
                        "ratio_all_samples": counts[bucket] / len(subset),
                    }
                )
    return output


def operation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for language in (*LANGUAGES, "overall"):
        subset = rows if language == "overall" else [row for row in rows if row["language"] == language]
        for operation in ("sub", "del", "ins"):
            b1_value = sum(int(row[f"{operation}_b1"]) for row in subset)
            m3_value = sum(int(row[f"{operation}_m3"]) for row in subset)
            output.append(
                {
                    "language": language,
                    "operation": operation,
                    "b1_count": b1_value,
                    "m3_count": m3_value,
                    "delta_m3_minus_b1": m3_value - b1_value,
                }
            )
    return output


def confusion_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        for model, alignment_key in (("b1", "_b1_alignment"), ("m3", "_m3_alignment")):
            for operation, gt_char, pred_char in row[alignment_key]:
                if operation == "sub":
                    counts[(row["language"], gt_char, pred_char)][model] += 1
    output = []
    for (language, gt_char, pred_char), model_counts in counts.items():
        b1_value = model_counts["b1"]
        m3_value = model_counts["m3"]
        output.append(
            {
                "language": language,
                "gt_char": gt_char,
                "pred_char": pred_char,
                "gt_codepoint": f"U+{ord(gt_char):04X}",
                "pred_codepoint": f"U+{ord(pred_char):04X}",
                "b1_count": b1_value,
                "m3_count": m3_value,
                "delta_m3_minus_b1": m3_value - b1_value,
                "reduction_ratio_vs_b1": (
                    (b1_value - m3_value) / b1_value if b1_value else ""
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            LANGUAGES.index(row["language"]),
            -abs(row["delta_m3_minus_b1"]),
            -row["b1_count"],
            row["gt_char"],
            row["pred_char"],
        ),
    )


def is_arabic_script(character: str) -> bool:
    return "ARABIC" in unicodedata.name(character, "")


def script_character_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = {
        "ug_arabic_script": ("ug", is_arabic_script),
        "kk_specific_cyrillic": ("kk", lambda char: char in KK_SPECIFIC),
    }
    output = []
    for category, (language, predicate) in categories.items():
        subset = [row for row in rows if row["language"] == language]
        reference_count = sum(sum(predicate(char) for char in row["gt"]) for row in subset)
        for model, alignment_key in (("B1", "_b1_alignment"), ("M3", "_m3_alignment")):
            substitutions = deletions = inserted = 0
            for row in subset:
                for operation, gt_char, pred_char in row[alignment_key]:
                    if operation == "sub" and predicate(gt_char):
                        substitutions += 1
                    elif operation == "del" and predicate(gt_char):
                        deletions += 1
                    elif operation == "ins" and predicate(pred_char):
                        inserted += 1
            errors = substitutions + deletions
            output.append(
                {
                    "category": category,
                    "language": language,
                    "model": model,
                    "reference_characters": reference_count,
                    "substitutions_on_reference_chars": substitutions,
                    "deletions_on_reference_chars": deletions,
                    "inserted_category_chars_reported_separately": inserted,
                    "reference_conditioned_error_count": errors,
                    "reference_conditioned_error_rate": errors / reference_count if reference_count else None,
                    "definition": "(substitutions + deletions) aligned to category GT characters / category GT characters; insertions are separate",
                }
            )
    return output


def exact_mcnemar(recovered: int, regressions: int) -> float:
    discordant = recovered + regressions
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(recovered, regressions) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def macro_cer_delta(rows: list[dict[str, Any]]) -> float:
    values = []
    for language in LANGUAGES:
        subset = [row for row in rows if row["language"] == language]
        denominator = sum(int(row["gt_len"]) for row in subset)
        values.append(
            (sum(int(row["ed_m3"]) for row in subset) - sum(int(row["ed_b1"]) for row in subset))
            / denominator
        )
    return statistics.fmean(values)


def paired_bootstrap(rows: list[dict[str, Any]], repetitions: int) -> dict[str, Any]:
    grouped = {language: [row for row in rows if row["language"] == language] for language in LANGUAGES}
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(repetitions):
        language_deltas = []
        for language in LANGUAGES:
            source = grouped[language]
            chosen = [source[rng.randrange(len(source))] for _ in range(len(source))]
            denominator = sum(int(row["gt_len"]) for row in chosen)
            delta = (
                sum(int(row["ed_m3"]) for row in chosen)
                - sum(int(row["ed_b1"]) for row in chosen)
            ) / denominator
            language_deltas.append(delta)
        samples.append(statistics.fmean(language_deltas))
    lower = percentile(samples, 0.025)
    upper = percentile(samples, 0.975)
    return {
        "estimand": "M3_macro_CER_minus_B1_macro_CER",
        "point_estimate": macro_cer_delta(rows),
        "confidence_level": 0.95,
        "ci_lower": lower,
        "ci_upper": upper,
        "repetitions": repetitions,
        "seed": BOOTSTRAP_SEED,
        "stratified_by_language": True,
        "paired_resampling": True,
        "ci_excludes_zero": upper < 0.0 or lower > 0.0,
    }


def create_or_read_ed1_review(output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = output / "ed1_manual_review.csv"
    candidates = [row for row in rows if int(row["ed_m3"]) == 1]
    rng = random.Random(BOOTSTRAP_SEED)
    selected = []
    for language in LANGUAGES:
        subset = [row for row in candidates if row["language"] == language]
        subset = sorted(subset, key=lambda row: row["sample_id"])
        rng.shuffle(subset)
        selected.extend(subset[:50])
    fields = [
        "sample_id", "language", "image_path", "absolute_image_path", "gt",
        "pred_b1", "pred_m3", "ed_b1", "ed_m3", "migration",
        "manual_decision", "manual_notes",
    ]
    if not path.exists():
        templates = []
        for row in selected:
            item = public_row(row)
            item["manual_decision"] = ""
            item["manual_notes"] = ""
            templates.append(item)
        write_csv(path, templates, fields)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reviewed = list(csv.DictReader(handle))
    expected = {(row["language"], row["sample_id"]) for row in selected}
    actual = {(row["language"], row["sample_id"]) for row in reviewed}
    if expected != actual:
        raise ValueError("ED=1 manual review membership changed; restore the generated template")
    decisions = Counter(row["manual_decision"].strip() for row in reviewed)
    invalid = sorted(decision for decision in decisions if decision and decision not in VALID_REVIEW_DECISIONS)
    if invalid:
        raise ValueError(f"Invalid ED=1 manual decisions: {invalid}")
    complete = bool(reviewed) and decisions[""] == 0
    genuine_ratio = decisions["genuine_ocr_error"] / len(reviewed) if complete else None
    reported_decisions = {
        (decision if decision else "unreviewed"): count
        for decision, count in decisions.items()
    }
    return {
        "path": str(path),
        "sample_count": len(reviewed),
        "complete": complete,
        "decision_counts": reported_decisions,
        "genuine_ocr_error_ratio": genuine_ratio,
        "genuine_error_majority_threshold": 0.5,
        "genuine_error_majority": genuine_ratio is not None and genuine_ratio >= 0.5,
    }


def create_recovered_reviews(output: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    paths = {}
    fields = [
        "sample_id", "language", "image_path", "absolute_image_path", "gt",
        "pred_b1", "pred_m3", "ed_b1", "ed_m3", "migration",
        "manual_category", "manual_notes",
    ]
    for language in LANGUAGES:
        path = output / f"recovered_{language}_review.csv"
        selected = [row for row in rows if row["language"] == language and row["migration"] == "WC"]
        templates = []
        for row in selected:
            item = public_row(row)
            item["manual_category"] = ""
            item["manual_notes"] = ""
            templates.append(item)
        if not path.exists():
            write_csv(path, templates, fields)
        paths[language] = str(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument(
        "--require-checkpoint-files",
        action="store_true",
        help="Require checkpoint files in addition to their frozen summary hashes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / "04_model_training/eval_reports" / PROTOCOL_ID).resolve()
    output.mkdir(parents=True, exist_ok=True)
    eval_root = root / "04_model_training/eval_reports"
    specifications = {
        "b1": {
            "summary": eval_root / "b1_full_s50_to_target_final_summary.json",
            "report": eval_root / "b1_full_s50_to_target_best_clean_dev",
        },
        "m3": {
            "summary": eval_root / "svtrv2_s_m3_dual_order_s50_to_target_final_summary.json",
            "report": eval_root / "svtrv2_s_m3_dual_order_s50_to_target_best_clean_dev",
        },
    }

    all_predictions = {}
    prediction_hashes = {}
    provenance = {}
    for model, spec in specifications.items():
        records, hashes = load_predictions(spec["report"], model)
        all_predictions[model] = records
        prediction_hashes.update(hashes)
        provenance[model] = validate_summary(
            root,
            spec["summary"],
            model,
            EXPECTED_EPOCHS[model],
            records,
            args.require_checkpoint_files,
        )

    if provenance["b1"]["summary"]["data_protocol_fingerprint"] != provenance["m3"]["summary"]["data_protocol_fingerprint"]:
        raise ValueError("B1/M3 data protocol fingerprint mismatch")
    method_freeze = root / "00_docs/frozen_method_design_v1/method_design_v1_manifest.json"

    paired = make_paired_rows(root, all_predictions["b1"], all_predictions["m3"])
    if len(paired) != 951 or Counter(row["language"] for row in paired) != Counter({"zh": 346, "ug": 295, "kk": 310}):
        raise ValueError("Frozen Clean-Dev membership/count mismatch")

    public = [public_row(row) for row in paired]
    write_csv(output / "error_migration_b1_vs_m3.csv", public)
    migrations = migration_summary(paired)
    write_csv(output / "error_migration_summary.csv", migrations)
    buckets = edit_distance_summary(paired)
    write_csv(output / "edit_distance_bucket.csv", buckets)
    operations = operation_summary(paired)
    write_csv(output / "error_operation_summary.csv", operations)
    confusions = confusion_summary(paired)
    write_csv(output / "character_confusion_delta.csv", confusions)
    script_chars = script_character_summary(paired)
    write_csv(output / "script_character_error_summary.csv", script_chars)

    recovered_regression = [public_row(row) for row in paired if row["migration"] in ("WC", "CW")]
    write_csv(output / "recovered_regression_samples.csv", recovered_regression)
    for migration, name in (("WC", "recovered"), ("CW", "regression")):
        for language in LANGUAGES:
            subset = [public_row(row) for row in paired if row["migration"] == migration and row["language"] == language]
            write_csv(output / f"{name}_{language}.csv", subset, list(public[0]))
    recovered_reviews = create_recovered_reviews(output, paired)

    overall_migration = next(row for row in migrations if row["language"] == "overall")
    mcnemar = {
        "test": "exact_two_sided_McNemar_binomial",
        "recovered_WC": overall_migration["recovered_WC"],
        "regression_CW": overall_migration["regression_CW"],
        "p_value": exact_mcnemar(
            overall_migration["recovered_WC"], overall_migration["regression_CW"]
        ),
    }
    bootstrap = paired_bootstrap(paired, args.bootstrap_repetitions)
    m3_errors = [row for row in paired if int(row["ed_m3"]) >= 1]
    m3_ed_counts = Counter(ed_bucket(int(row["ed_m3"])) for row in paired)
    ed1_ratio = m3_ed_counts["ED=1"] / len(m3_errors) if m3_errors else 0.0
    quantity_condition_a = ed1_ratio >= 0.5
    quantity_condition_b = (
        m3_ed_counts["ED=1"] > m3_ed_counts["ED=2"]
        and m3_ed_counts["ED=1"] > m3_ed_counts["ED>=3"]
    )
    manual_review = create_or_read_ed1_review(output, paired)
    quantitative_candidate = quantity_condition_a or quantity_condition_b
    if not manual_review["complete"]:
        hem_decision = "pending_manual_ed1_review"
    elif quantitative_candidate and manual_review["genuine_error_majority"]:
        hem_decision = "go"
    else:
        hem_decision = "no_go"

    source_metrics = {}
    for model in ("b1", "m3"):
        dev = provenance[model]["summary"]["dev"]
        source_metrics[model.upper()] = {
            "epoch": EXPECTED_EPOCHS[model],
            "macro_cer": dev["macro_cer"],
            "macro_wer": statistics.fmean(dev["languages"][language]["wer"] for language in LANGUAGES),
            "macro_1_ned": statistics.fmean(dev["languages"][language]["one_minus_ned_macro"] for language in LANGUAGES),
            "macro_line_accuracy": dev["macro_line_accuracy"],
        }
    source_metrics["M3"]["alpha"] = EXPECTED_ALPHA

    ug_migration = next(row for row in migrations if row["language"] == "ug")
    kk_regressions = [row for row in paired if row["language"] == "kk" and row["migration"] == "CW"]
    kk_regressions_ed1 = bool(kk_regressions) and all(int(row["ed_m3"]) == 1 for row in kk_regressions)
    decision_rows = [
        {"criterion": "M3 Macro CER < B1", "result": source_metrics["M3"]["macro_cer"] < source_metrics["B1"]["macro_cer"]},
        {"criterion": "M3 Macro Line Accuracy > B1", "result": source_metrics["M3"]["macro_line_accuracy"] > source_metrics["B1"]["macro_line_accuracy"]},
        {"criterion": "Overall Recovered > Regression", "result": overall_migration["recovered_WC"] > overall_migration["regression_CW"]},
        {"criterion": "Uyghur Recovered > Regression", "result": ug_migration["recovered_WC"] > ug_migration["regression_CW"]},
        {"criterion": "All Kazakh regressions are ED=1", "result": kk_regressions_ed1},
        {"criterion": "ED=1 share of remaining errors >= 50%", "result": quantity_condition_a},
        {"criterion": "ED=1 manual review complete", "result": manual_review["complete"]},
        {"criterion": "ED=1 majority are genuine OCR errors", "result": manual_review["genuine_error_majority"] if manual_review["complete"] else "PENDING"},
        {"criterion": "Proceed to HEM", "result": hem_decision},
    ]
    write_csv(output / "stage6_decision_table.csv", decision_rows)

    summary = {
        "status": "STAGE6_ERROR_MIGRATION_COMPLETE",
        "protocol_id": PROTOCOL_ID,
        "analysis_only": True,
        "training_performed": False,
        "inference_performed": False,
        "gpu_policy": "CUDA_VISIBLE_DEVICES=0 enforced by tmux wrapper; reused frozen predictions, so no GPU is expected",
        "test_evaluated": False,
        "corrupted_dev_used": False,
        "source_metrics": source_metrics,
        "provenance": provenance,
        "prediction_sha256": prediction_hashes,
        "method_freeze_present": method_freeze.is_file(),
        "method_freeze_sha256": sha256(method_freeze) if method_freeze.is_file() else None,
        "alignment_tie_breaking": "match, then substitution, deletion, insertion",
        "migration": migrations,
        "delta_ed": {
            "mean_m3_minus_b1": statistics.fmean(row["delta_ed_m3_minus_b1"] for row in paired),
            "improved_samples": sum(row["delta_ed_m3_minus_b1"] < 0 for row in paired),
            "unchanged_samples": sum(row["delta_ed_m3_minus_b1"] == 0 for row in paired),
            "worsened_samples": sum(row["delta_ed_m3_minus_b1"] > 0 for row in paired),
        },
        "significance": {"line_accuracy_mcnemar": mcnemar, "macro_cer_paired_bootstrap": bootstrap},
        "hem_gate": {
            "m3_remaining_error_rows": len(m3_errors),
            "m3_ed1_rows": m3_ed_counts["ED=1"],
            "ed1_share_of_remaining_errors": ed1_ratio,
            "condition_a_ed1_share_at_least_50_percent": quantity_condition_a,
            "condition_b_ed1_largest_individual_error_bucket": quantity_condition_b,
            "quantitative_candidate": quantitative_candidate,
            "manual_review": manual_review,
            "decision": hem_decision,
        },
        "manual_recovered_review_files": recovered_reviews,
        "outputs": {
            "paired_samples": str(output / "error_migration_b1_vs_m3.csv"),
            "migration": str(output / "error_migration_summary.csv"),
            "ed_buckets": str(output / "edit_distance_bucket.csv"),
            "operations": str(output / "error_operation_summary.csv"),
            "confusions": str(output / "character_confusion_delta.csv"),
            "script_characters": str(output / "script_character_error_summary.csv"),
            "recovered_regression": str(output / "recovered_regression_samples.csv"),
            "summary": str(output / "stage6_summary.json"),
            "decision_table": str(output / "stage6_decision_table.csv"),
        },
    }
    write_json(output / "stage6_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("STAGE6_ERROR_MIGRATION_COMPLETE; no training, Corrupted Dev, or Test was used.", flush=True)


if __name__ == "__main__":
    main()
