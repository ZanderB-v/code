#!/usr/bin/env python3
"""Read-only SCDL mechanism audit on a fixed sample of target Train batches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_NAMES = ("Han", "Arabic", "Cyrillic")
PROTOCOL_ID = "SCDL_TARGET_TRAIN_INTERNAL_DIAGNOSTIC_V2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--negative-examples-per-script", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def reservoir_add(
    samples: list[dict], candidate: dict, seen: int, limit: int, rng: random.Random
) -> None:
    if len(samples) < limit:
        samples.append(candidate)
    else:
        index = rng.randrange(seen)
        if index < limit:
            samples[index] = candidate


def prototype_coverage(counts, script_ids) -> dict:
    result = {}
    for index, name in enumerate(SCRIPT_NAMES):
        values = counts[script_ids == index]
        result[name] = {
            "dictionary_characters": int(values.numel()),
            "covered_characters": int((values > 0).sum()),
            "coverage_ratio": float((values > 0).float().mean()) if values.numel() else None,
            "count_0": int((values == 0).sum()),
            "count_1_to_9": int(((values >= 1) & (values < 10)).sum()),
            "count_10_to_49": int(((values >= 10) & (values < 50)).sum()),
            "count_50_plus": int((values >= 50).sum()),
            "median_token_observations": float(values.median()) if values.numel() else None,
        }
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing audit: {output}")
    if args.max_batches < 1 or args.batch_size < 1:
        raise ValueError("Batch count and batch size must be positive")
    if args.negative_examples_per_script < 1:
        raise ValueError("Negative-example sample count must be positive")
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("SCDL target config/checkpoint is missing")

    openocr = root / "third_party/OpenOCR"
    sys.path.insert(0, str(openocr))
    sys.path.insert(0, str(openocr / "tools"))
    import torch
    import torch.nn.functional as F
    from openrec.losses import build_loss
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from tools.data import build_dataloader
    from tools.engine.config import Config
    from tools.utils.logging import get_logger

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose only physical GPU0 with CUDA_VISIBLE_DEVICES=0")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    cfg = Config(str(config_path)).cfg
    if cfg["Global"].get("method_variant") != "m3_scdl_full_v1":
        raise ValueError("This audit requires the frozen M3+SCDL full V1 config")
    if not cfg["Architecture"]["Decoder"].get("use_scdl"):
        raise ValueError("SCDL is disabled in config")
    loss_cfg = cfg["Loss"]
    expected = {
        "consistency_weight": 0.15,
        "scdl_weight": 0.05,
        "scdl_topk": 5,
        "scdl_temperature": 0.1,
        "scdl_warmup_fraction": 0.2,
    }
    for key, value in expected.items():
        if not math.isclose(float(loss_cfg[key]), value, abs_tol=1e-9):
            raise ValueError(f"Frozen SCDL config mismatch: {key}")
    sources = cfg["Train"]["dataset"].get("data_dir_list", [])
    if not sources or any(
        {"dev", "test"} & {part.lower() for part in Path(source).parts}
        for source in sources
    ):
        raise ValueError("Only target Train LMDB sources are allowed")
    for source in sources:
        if not Path(source).is_dir():
            raise FileNotFoundError(source)
    cfg["Global"]["distributed"] = False
    cfg["Global"]["use_amp"] = False
    cfg["Train"]["loader"]["num_workers"] = 0
    cfg["Train"]["loader"]["batch_size_per_card"] = args.batch_size
    cfg["Train"]["sampler"].pop("control_world_size", None)
    cfg["Train"]["sampler"].pop("control_first_bs", None)
    cfg["Train"]["sampler"]["first_bs"] = args.batch_size
    logger = get_logger("scdl_internal_diagnostic")
    postprocess = build_post_process(cfg["PostProcess"], cfg["Global"])
    cfg["Architecture"]["Decoder"]["out_channels"] = (
        postprocess.get_character_num()
    )
    dataloader = build_dataloader(
        cfg, "Train", logger, seed=args.seed, epoch=1, task="rec"
    )
    device = torch.device("cuda:0")
    model = build_model(cfg["Architecture"]).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in payload or "epoch" not in payload:
        raise ValueError("Expected a complete selected training checkpoint")
    model.load_state_dict(payload["state_dict"], strict=True)
    bank_module = model.decoder.scdl
    counts = bank_module.prototype_counts.detach().float().cpu()
    prototype_scripts = bank_module.prototype_script_ids.detach().cpu()
    if int((counts > 0).sum()) == 0:
        raise ValueError("Selected SCDL checkpoint has an empty prototype bank")
    coverage = prototype_coverage(counts, prototype_scripts)
    dictionary = Path(cfg["Architecture"]["Decoder"]["scdl_character_dict_path"])
    characters = dictionary.read_text(encoding="utf-8-sig").splitlines()
    if cfg["Architecture"]["Decoder"].get("scdl_use_space_char", True):
        characters.append(" ")
    if len(characters) != counts.numel():
        raise ValueError("Dictionary and prototype bank differ")
    loss_fn = build_loss(loss_cfg)
    # Training mode exposes SCDL frame features. No loss.forward(), backward(),
    # optimizer step, save, or bank update is performed in this read-only audit.
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    measurements = defaultdict(lambda: defaultdict(list))
    negative_examples = {name: [] for name in SCRIPT_NAMES}
    seen_negative_examples = Counter()
    rng = random.Random(args.seed)
    sampled_batches = 0
    valid_alignment_samples = 0
    total_samples = 0
    eligible_tokens = Counter()
    usable_tokens = Counter()
    with torch.no_grad():
        for batch in dataloader:
            if sampled_batches >= args.max_batches:
                break
            batch = [item.to(device) for item in batch]
            predictions = model(batch[0], data=batch[1:])
            required = (
                "ctc_pred", "scdl_frame_features", "scdl_prototypes",
                "scdl_prototype_counts", "scdl_prototype_script_ids",
            )
            if any(key not in predictions for key in required):
                raise ValueError("SCDL training outputs missing from checkpoint model")
            lengths = batch[8].long()
            visual_ids = batch[9].long()
            visual_scripts = batch[13].long()
            gamma, valid_samples = loss_fn._ctc_token_posteriors(
                predictions["ctc_pred"], visual_ids, lengths
            )
            valid_alignment_samples += int(valid_samples.sum())
            total_samples += len(lengths)
            frames = predictions["scdl_frame_features"]
            pooled = gamma @ frames
            pooled = F.normalize(
                pooled / gamma.sum(dim=-1, keepdim=True).clamp_min(1e-7),
                dim=-1,
            )
            frame_prob = F.softmax(predictions["ctc_pred"].float(), dim=-1)
            bank = predictions["scdl_prototypes"].detach()
            bank_counts = predictions["scdl_prototype_counts"].detach()
            bank_scripts = predictions["scdl_prototype_script_ids"].detach()
            max_tokens = gamma.shape[1]
            token_positions = torch.arange(max_tokens, device=device)[None, :]
            valid = (
                (token_positions < lengths[:, None])
                & valid_samples[:, None]
                & (visual_ids[:, :max_tokens] > 0)
                & (visual_scripts[:, :max_tokens] >= 0)
                & (visual_scripts[:, :max_tokens] < 3)
            )
            features = pooled[valid]
            labels = visual_ids[:, :max_tokens][valid] - 1
            scripts = visual_scripts[:, :max_tokens][valid]
            token_gamma = gamma[valid]
            sample_indices = torch.arange(len(lengths), device=device)[:, None]
            sample_indices = sample_indices.expand(-1, max_tokens)[valid]
            # Prototype IDs are zero-based over characters; CTC class 0 is
            # blank, so the matching character logit is prototype ID + 1.
            ctc_class_ids = labels + 1
            if (ctc_class_ids <= 0).any() or (
                ctc_class_ids >= frame_prob.shape[-1]
            ).any():
                raise ValueError("Prototype-to-CTC class mapping is invalid")
            target_frame_prob = frame_prob[sample_indices, :, ctc_class_ids]
            normalized_gamma = token_gamma / token_gamma.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-7)
            target_confidence = (
                normalized_gamma * target_frame_prob
            ).sum(dim=-1)
            concentration = normalized_gamma.max(dim=-1).values
            for script_id, name in enumerate(SCRIPT_NAMES):
                selected = scripts == script_id
                eligible_tokens[name] += int(selected.sum())
                if not selected.any():
                    continue
                mask = (bank_scripts == script_id) & (bank_counts > 0)
                candidate_ids = mask.nonzero(as_tuple=False).squeeze(1)
                if candidate_ids.numel() < 2:
                    continue
                f = features[selected]
                y = labels[selected]
                confidences = target_confidence[selected]
                peaks = concentration[selected]
                ready = bank_counts[y] > 0
                f, y = f[ready], y[ready]
                confidences, peaks = confidences[ready], peaks[ready]
                if f.numel() == 0:
                    continue
                usable_tokens[name] += len(y)
                similarity = f @ bank[candidate_ids].T
                similarity = similarity.masked_fill(
                    y[:, None] == candidate_ids[None, :], float("-inf")
                )
                topk = min(int(loss_fn.scdl_topk), candidate_ids.numel() - 1)
                hard_values, hard_indices = similarity.topk(topk, dim=1)
                positive = (f * bank[y]).sum(dim=-1)
                logits = torch.cat((positive[:, None], hard_values), dim=1)
                per_token_loss = F.cross_entropy(
                    logits / loss_fn.scdl_temperature,
                    torch.zeros(len(y), dtype=torch.long, device=device),
                    reduction="none",
                )
                for key, tensor in (
                    ("scdl_loss", per_token_loss),
                    ("ctc_target_probability", confidences),
                    ("alignment_peak_concentration", peaks),
                    ("positive_similarity", positive),
                    ("topk_negative_similarity", hard_values.mean(dim=1)),
                ):
                    measurements[name][key].extend(tensor.cpu().tolist())
                for index in range(len(y)):
                    seen_negative_examples[name] += 1
                    label_id = int(y[index])
                    hard_ids = candidate_ids[hard_indices[index]].tolist()
                    candidate = {
                        "script": name,
                        "positive_character": characters[label_id],
                        "negative_characters": " | ".join(
                            characters[int(item)] for item in hard_ids
                        ),
                        "positive_similarity": float(positive[index]),
                        "mean_topk_negative_similarity": float(
                            hard_values[index].mean()
                        ),
                        "ctc_target_probability": float(confidences[index]),
                        "alignment_peak_concentration": float(peaks[index]),
                    }
                    reservoir_add(
                        negative_examples[name], candidate,
                        seen_negative_examples[name],
                        args.negative_examples_per_script, rng,
                    )
            sampled_batches += 1
            if sampled_batches % 8 == 0:
                print(f"sampled_train_batches={sampled_batches}", flush=True)

    if sampled_batches == 0:
        raise RuntimeError("No target Train batches were sampled")
    diagnostics = []
    for name in SCRIPT_NAMES:
        row = {
            "script": name,
            "eligible_tokens": eligible_tokens[name],
            "usable_tokens": usable_tokens[name],
            "mean_scdl_loss": mean(measurements[name]["scdl_loss"]),
            "mean_ctc_target_probability": mean(
                measurements[name]["ctc_target_probability"]
            ),
            "mean_alignment_peak_concentration": mean(
                measurements[name]["alignment_peak_concentration"]
            ),
            "mean_positive_similarity": mean(
                measurements[name]["positive_similarity"]
            ),
            "mean_topk_negative_similarity": mean(
                measurements[name]["topk_negative_similarity"]
            ),
            **coverage[name],
        }
        diagnostics.append(row)
    if any(row["usable_tokens"] == 0 for row in diagnostics):
        raise RuntimeError("At least one script has no usable diagnostic tokens")
    output.mkdir(parents=True)
    write_csv(output / "script_diagnostics.csv", diagnostics, list(diagnostics[0]))
    examples = [row for name in SCRIPT_NAMES for row in negative_examples[name]]
    write_csv(output / "hard_negative_examples.csv", examples, list(examples[0]))
    summary = {
        "status": "SCDL_INTERNAL_DIAGNOSTIC_COMPLETE",
        "protocol_id": PROTOCOL_ID,
        "data_split": "target_train_only",
        "physical_gpu": 0,
        "checkpoint_epoch": int(payload["epoch"]),
        "sampled_batches": sampled_batches,
        "sampled_lines": total_samples,
        "ctc_alignment_valid_lines": valid_alignment_samples,
        "script_diagnostics": diagnostics,
        "hard_negative_examples_per_script": {
            name: len(negative_examples[name]) for name in SCRIPT_NAMES
        },
        "provenance": {
            "checkpoint_sha256": sha256(checkpoint_path),
            "config_sha256": sha256(config_path),
            "random_seed": args.seed,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "prototype_counts_are_token_observations_not_unique_samples": True,
            "checkpoint_bank_frozen_during_diagnostic": True,
            "ctc_class_index_offset_from_prototype": 1,
        },
        "training_run": False,
        "corrupted_dev_used": False,
        "test_evaluated": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
