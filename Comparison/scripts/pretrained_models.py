#!/usr/bin/env python3
"""Native public-baseline models and fail-closed checkpoint transfer rules."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import math
import sys
import types
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_ROOT = PROJECT_ROOT / "Comparison"
PARSEQ_SOURCE = COMPARISON_ROOT / "third_party" / "parseq_v1_0_0"
MMOCR_SOURCE = COMPARISON_ROOT / "third_party" / "mmocr_v1_0_1"


def prepend_source_paths() -> None:
    for path in (PARSEQ_SOURCE, MMOCR_SOURCE):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


class _NoOpRegistry:
    """Minimal registry used when loading isolated MMOCR recognition sources."""

    def register_module(self, *args, **kwargs):
        def decorator(module):
            return module

        return decorator


def _ensure_package(name: str, path: Path | None = None) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [] if path is None else [str(path)]
        sys.modules[name] = module
    return module


def _load_source_module(name: str, path: Path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_isolated_mmocr_recognition_modules():
    """Load only MMOCR's STN and SVTR sources, without detection extensions.

    MMOCR's package-level imports also load detection modules backed by compiled
    ``mmcv.ops``.  The public SVTR recognizer uses only ``mmcv.cnn`` and does not
    require those operators, so loading its two source files in an isolated
    package keeps the implementation exact while avoiding an unrelated runtime
    dependency on ``mmcv-full``.
    """

    source_root = MMOCR_SOURCE / "mmocr"
    fake_mmocr = _ensure_package("mmocr", source_root)

    registry = types.ModuleType("mmocr.registry")
    registry.MODELS = _NoOpRegistry()
    sys.modules["mmocr.registry"] = registry
    fake_mmocr.registry = registry

    structures = types.ModuleType("mmocr.structures")
    structures.TextRecogDataSample = type("TextRecogDataSample", (), {})
    sys.modules["mmocr.structures"] = structures
    fake_mmocr.structures = structures

    private_root = "_comparison_mmocr_recognition"
    _ensure_package(private_root)
    _ensure_package(f"{private_root}.preprocessors")
    _ensure_package(f"{private_root}.encoders")

    preprocessor_dir = source_root / "models" / "textrecog" / "preprocessors"
    _load_source_module(
        f"{private_root}.preprocessors.base", preprocessor_dir / "base.py"
    )
    preprocessor_module = _load_source_module(
        f"{private_root}.preprocessors.tps_preprocessor",
        preprocessor_dir / "tps_preprocessor.py",
    )
    encoder_module = _load_source_module(
        f"{private_root}.encoders.svtr_encoder",
        source_root / "models" / "textrecog" / "encoders" / "svtr_encoder.py",
    )
    return preprocessor_module, encoder_module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_state(path: Path) -> OrderedDict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                payload = candidate
                break
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint payload in {path}: {type(payload)!r}")
    state = OrderedDict()
    for key, value in payload.items():
        if torch.is_tensor(value):
            state[str(key)] = value
    if not state:
        raise ValueError(f"No tensors found in checkpoint: {path}")
    return state


class PARSeqTransfer(nn.Module):
    """PARSeq v1.0.0 body with a project-specific vocabulary and length."""

    def __init__(self, dictionary_size: int, max_label_length: int) -> None:
        super().__init__()
        prepend_source_paths()
        modules = importlib.import_module("strhub.models.parseq.modules")
        self.encoder = modules.Encoder(
            img_size=(32, 128),
            patch_size=(4, 8),
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
        )
        decoder_layer = modules.DecoderLayer(384, 12, 384 * 4, 0.1)
        self.decoder = modules.Decoder(
            decoder_layer, num_layers=1, norm=nn.LayerNorm(384)
        )
        self.head = nn.Linear(384, dictionary_size + 1)
        self.text_embed = modules.TokenEmbedding(dictionary_size + 3, 384)
        self.pos_queries = nn.Parameter(torch.empty(1, max_label_length + 1, 384))
        nn.init.trunc_normal_(self.pos_queries, std=0.02)
        self.dropout = nn.Dropout(0.1)
        self.max_label_length = max_label_length
        self.eos_id = 0
        self.bos_id = dictionary_size + 1
        self.pad_id = dictionary_size + 2
        self.rng = np.random.default_rng(20260815)

    def decode(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask=None,
        target_padding_mask=None,
        target_query=None,
        target_query_mask=None,
    ) -> torch.Tensor:
        batch, length = target.shape
        null_context = self.text_embed(target[:, :1])
        target_embedding = self.pos_queries[:, : length - 1] + self.text_embed(
            target[:, 1:]
        )
        target_embedding = self.dropout(
            torch.cat([null_context, target_embedding], dim=1)
        )
        if target_query is None:
            target_query = self.pos_queries[:, :length].expand(batch, -1, -1)
        return self.decoder(
            self.dropout(target_query),
            target_embedding,
            memory,
            target_query_mask,
            target_mask,
            target_padding_mask,
        )

    def forward(
        self,
        images: torch.Tensor,
        token_rows: list[list[int]] | None = None,
    ) -> torch.Tensor:
        # Keeping both paths behind ``forward`` is required for correct DDP
        # reducer bookkeeping; custom methods must not bypass DDP.forward.
        if token_rows is not None:
            loss, _ = self.training_loss(images, token_rows)
            return loss
        return self.forward_inference(images)

    def _build_targets(self, token_rows: list[list[int]]) -> torch.Tensor:
        max_chars = max(len(row) for row in token_rows)
        if max_chars > self.max_label_length:
            raise ValueError(
                f"PARSeq label length {max_chars} exceeds capacity "
                f"{self.max_label_length}; truncation is forbidden"
            )
        target = torch.full(
            (len(token_rows), max_chars + 2),
            self.pad_id,
            dtype=torch.long,
            device=self.pos_queries.device,
        )
        target[:, 0] = self.bos_id
        for index, row in enumerate(token_rows):
            row = row[:max_chars]
            if row:
                target[index, 1 : len(row) + 1] = torch.tensor(
                    row, dtype=torch.long, device=target.device
                )
            target[index, len(row) + 1] = self.eos_id
        return target

    def _generate_target_permutations(self, target: torch.Tensor) -> torch.Tensor:
        """Released PARSeq v1.0.0 recipe: six mirrored permutations."""

        max_num_chars = target.shape[1] - 2
        if max_num_chars == 1:
            return torch.arange(3, device=target.device).unsqueeze(0)
        generated = [torch.arange(max_num_chars, device=target.device)]
        max_permutations = math.factorial(max_num_chars) // 2
        generated_count = min(3, max_permutations)
        if max_num_chars < 5:
            selector = (
                [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
                if max_num_chars == 4
                else list(range(max_permutations))
            )
            pool = torch.as_tensor(
                list(permutations(range(max_num_chars), max_num_chars)),
                device=target.device,
            )[selector][1:]
            stacked = torch.stack(generated)
            if len(pool):
                indices = self.rng.choice(
                    len(pool), size=generated_count - len(stacked), replace=False
                )
                stacked = torch.cat([stacked, pool[indices]])
        else:
            generated.extend(
                torch.randperm(max_num_chars, device=target.device)
                for _ in range(generated_count - len(generated))
            )
            stacked = torch.stack(generated)
        mirrored = stacked.flip(-1)
        stacked = torch.stack([stacked, mirrored]).transpose(0, 1).reshape(
            -1, max_num_chars
        )
        bos = stacked.new_zeros((len(stacked), 1))
        eos = stacked.new_full((len(stacked), 1), max_num_chars + 1)
        stacked = torch.cat([bos, stacked + 1, eos], dim=1)
        if len(stacked) > 1:
            stacked[1, 1:] = max_num_chars + 1 - torch.arange(
                max_num_chars + 1, device=target.device
            )
        return stacked

    @staticmethod
    def _generate_attention_masks(permutation: torch.Tensor):
        size = permutation.shape[0]
        mask = torch.zeros((size, size), device=permutation.device)
        for index in range(size):
            query_index = permutation[index]
            mask[query_index, permutation[index + 1 :]] = float("-inf")
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(size, dtype=torch.bool, device=permutation.device)] = float(
            "-inf"
        )
        return content_mask, mask[1:, :-1]

    def training_loss(
        self, images: torch.Tensor, token_rows: list[list[int]]
    ) -> tuple[torch.Tensor, list[int]]:
        target = self._build_targets(token_rows)
        memory = self.encoder(images)
        target_input = target[:, :-1]
        target_output = target[:, 1:]
        padding_mask = (target_input == self.pad_id) | (
            target_input == self.eos_id
        )
        loss = images.new_zeros(())
        loss_elements = 0
        last_logits = None
        for index, permutation in enumerate(
            self._generate_target_permutations(target)
        ):
            target_mask, query_mask = self._generate_attention_masks(permutation)
            output = self.decode(
                target_input,
                memory,
                target_mask,
                padding_mask,
                target_query_mask=query_mask,
            )
            logits = self.head(output)
            elements = int((target_output != self.pad_id).sum().item())
            if elements:
                loss = loss + elements * nn.functional.cross_entropy(
                    logits.flatten(end_dim=1),
                    target_output.flatten(),
                    ignore_index=self.pad_id,
                )
                loss_elements += elements
            last_logits = logits
            if index == 1:
                target_output = torch.where(
                    target_output == self.eos_id,
                    self.pad_id,
                    target_output,
                )
        if not loss_elements or last_logits is None:
            raise ValueError("PARSeq permutation objective has no supervised tokens")
        return loss / loss_elements, list(last_logits.shape)

    def training_logits(self, images: torch.Tensor, token_rows: list[list[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = len(token_rows)
        max_chars = max(len(row) for row in token_rows)
        if max_chars > self.max_label_length:
            raise ValueError(
                f"PARSeq label length {max_chars} exceeds capacity "
                f"{self.max_label_length}; truncation is forbidden"
            )
        steps = max_chars + 1
        tgt_in = torch.full(
            (batch, steps), self.pad_id, dtype=torch.long, device=images.device
        )
        target = torch.full(
            (batch, steps), -100, dtype=torch.long, device=images.device
        )
        tgt_in[:, 0] = self.bos_id
        for index, row in enumerate(token_rows):
            row = row[:max_chars]
            if row:
                values = torch.tensor(row, dtype=torch.long, device=images.device)
                tgt_in[index, 1 : len(row) + 1] = values
                target[index, : len(row)] = values
            target[index, len(row)] = self.eos_id

        memory = self.encoder(images)
        null_context = self.text_embed(tgt_in[:, :1])
        content = self.pos_queries[:, : steps - 1] + self.text_embed(tgt_in[:, 1:])
        content = torch.cat([null_context, content], dim=1)
        query = self.pos_queries[:, :steps].expand(batch, -1, -1)
        causal = torch.triu(
            torch.full((steps, steps), float("-inf"), device=images.device), 1
        )
        padding = tgt_in.eq(self.pad_id)
        decoded = self.decoder(
            query,
            content,
            memory,
            query_mask=causal,
            content_mask=causal,
            content_key_padding_mask=padding,
        )
        return self.head(decoded), target

    def forward_inference(self, images: torch.Tensor) -> torch.Tensor:
        """Released PARSeq AR decode plus one cloze refinement iteration."""

        batch = images.shape[0]
        steps = self.max_label_length + 1
        memory = self.encoder(images)
        positions = self.pos_queries[:, :steps].expand(batch, -1, -1)
        causal = torch.triu(
            torch.full((steps, steps), float("-inf"), device=images.device), 1
        )
        target = torch.full(
            (batch, steps), self.pad_id, dtype=torch.long, device=images.device
        )
        target[:, 0] = self.bos_id
        outputs = []
        for index in range(steps):
            length = index + 1
            decoded = self.decode(
                target[:, :length],
                memory,
                causal[:length, :length],
                target_query=positions[:, index:length],
                target_query_mask=causal[index:length, :length],
            )
            logits = self.head(decoded)
            outputs.append(logits)
            if length < steps:
                target[:, length] = logits.squeeze(1).argmax(-1)
                if (target == self.eos_id).any(dim=-1).all():
                    break
        logits = torch.cat(outputs, dim=1)

        # The released public PARSeq checkpoint uses one refinement iteration.
        decoded_steps = logits.shape[1]
        refine_mask = causal[:decoded_steps, :decoded_steps].clone()
        upper_two = torch.triu(
            torch.ones(
                decoded_steps,
                decoded_steps,
                dtype=torch.bool,
                device=images.device,
            ),
            2,
        )
        refine_mask[upper_two] = 0
        bos = torch.full(
            (batch, 1), self.bos_id, dtype=torch.long, device=images.device
        )
        target = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
        padding = (target == self.eos_id).cumsum(-1) > 0
        decoded = self.decode(
            target,
            memory,
            refine_mask,
            padding,
            target_query=positions[:, :decoded_steps],
            target_query_mask=refine_mask[:, : target.shape[1]],
        )
        return self.head(decoded)


class MinimalSVTRDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.decoder = nn.Linear(in_channels, out_channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[2] != 1:
            raise ValueError(f"Unexpected SVTR feature shape: {tuple(feature.shape)}")
        return self.decoder(feature.squeeze(2).permute(0, 2, 1))


class SVTRTransfer(nn.Module):
    """MMOCR v1.0.1 SVTR-Base with only its vocabulary head replaced."""

    def __init__(self, dictionary_size: int, sequence_capacity: int) -> None:
        super().__init__()
        prepend_source_paths()
        try:
            preprocessor_module, encoder_module = (
                load_isolated_mmocr_recognition_modules()
            )
        except Exception as exc:  # pragma: no cover - exercised on the server
            raise RuntimeError(
                "SVTR requires the bundled MMOCR runtime. Run "
                "Comparison/scripts/install_pretrained_baseline_runtime.sh first."
            ) from exc
        self.preprocessor = preprocessor_module.STN(
            in_channels=3,
            resized_image_size=(32, 64),
            output_image_size=(48, 160),
            num_control_points=20,
            margins=[0.05, 0.05],
        )
        self.encoder = encoder_module.SVTREncoder(
            img_size=[48, 160],
            in_channels=3,
            out_channels=256,
            embed_dims=[128, 256, 384],
            depth=[3, 6, 9],
            num_heads=[4, 8, 12],
            mixer_types=["Local"] * 8 + ["Global"] * 10,
            window_size=[[7, 11], [7, 11], [7, 11]],
            merging_types="Conv",
            prenorm=False,
            max_seq_len=sequence_capacity,
        )
        self.decoder = MinimalSVTRDecoder(256, dictionary_size + 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # MMOCR's TPS/STN contains a dense grid_sample and precomputed TPS
        # coordinate matrices.  Running that block under CUDA FP16 can create
        # non-finite gradients even when its forward output is finite.  Keep
        # only the geometric rectification in FP32; the SVTR encoder and the
        # replacement head remain inside the caller's AMP context.
        autocast_off = (
            torch.autocast(device_type=images.device.type, enabled=False)
            if images.is_cuda
            else contextlib.nullcontext()
        )
        with autocast_off:
            rectified = self.preprocessor(images.float())
        return self.decoder(self.encoder(rectified))


@dataclass
class ModelBundle:
    name: str
    model: nn.Module
    source_state: OrderedDict[str, torch.Tensor]
    reset_key: Callable[[str], bool]
    backbone_key: Callable[[str], bool]
    preserved_body_key: Callable[[str], bool]
    input_size: Tuple[int, int]
    objective: str
    reset_modules: list[str]
    runtime_compatibility_notes: list[str] = field(default_factory=list)


def apply_abinet_transformer_decoder_compatibility(model: nn.Module) -> list[str]:
    """Preserve the decoder semantics expected by the released ABINet code.

    ABINet v1.0.0 supplies a custom decoder layer whose forward signature
    predates the causal-mask arguments added to PyTorch's TransformerDecoder.
    PyTorch 2.x also inspects ``layer.self_attn.batch_first`` even when the
    released ABINet layer intentionally disables self-attention.  Binding the
    legacy container loop avoids both framework-level assumptions without
    changing modules, parameters, state-dict keys, or tensor operations.
    """

    decoder = model.language.model

    def released_forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output

    decoder.forward = types.MethodType(released_forward, decoder)
    return [
        "ABINet v1.0.0 TransformerDecoder container uses released pre-causal-mask "
        "semantics for PyTorch 2.x compatibility; modules, parameters, state-dict "
        "keys, and tensor operations are unchanged."
    ]


def _strip_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> OrderedDict[str, torch.Tensor]:
    output = OrderedDict()
    for key, value in state.items():
        output[key[len(prefix) :] if key.startswith(prefix) else key] = value
    return output


def build_model_bundle(
    name: str,
    checkpoint: Path,
    dictionary_size: int,
    max_label_length: int,
) -> ModelBundle:
    prepend_source_paths()
    source = load_checkpoint_state(checkpoint)

    if name == "crnn":
        module = importlib.import_module("strhub.models.crnn.model")
        model = module.CRNN(32, 3, dictionary_size + 1, 256, False)
        source = _strip_prefix(source, "model.")
        reset = lambda key: key.startswith("rnn.1.linear.")
        backbone = lambda key: key.startswith("cnn.") or key.startswith("rnn.0.") or key.startswith("rnn.1.rnn.")
        return ModelBundle(
            name,
            model,
            source,
            reset,
            backbone,
            backbone,
            (32, 512),
            "ctc",
            ["rnn.1.linear"],
        )

    if name == "parseq":
        model = PARSeqTransfer(dictionary_size, max_label_length)
        reset = lambda key: key == "pos_queries" or key.startswith("head.") or key.startswith("text_embed.")
        backbone = lambda key: key.startswith("encoder.")
        preserved = lambda key: backbone(key) or key.startswith("decoder.")
        return ModelBundle(
            name,
            model,
            source,
            reset,
            backbone,
            preserved,
            (32, 128),
            "parseq",
            ["head", "text_embed", "pos_queries"],
        )

    if name == "abinet":
        module = importlib.import_module("strhub.models.abinet.model_abinet_iter")
        model = module.ABINetIterModel(
            max_label_length,
            0,
            dictionary_size + 1,
            iter_size=3,
            d_model=512,
            nhead=8,
            d_inner=2048,
            dropout=0.1,
            activation="relu",
            v_loss_weight=1.0,
            v_attention="position",
            v_attention_mode="nearest",
            v_backbone="transformer",
            v_num_layers=3,
            l_loss_weight=1.0,
            l_num_layers=4,
            l_detach=True,
            l_use_self_attn=False,
            a_loss_weight=1.0,
        )
        compatibility_notes = apply_abinet_transformer_decoder_compatibility(model)
        source = _strip_prefix(source, "model.")

        def reset(key: str) -> bool:
            vocabulary_modules = (
                "vision.cls.",
                "language.proj.",
                "language.cls.",
                "alignment.cls.",
            )
            positional_buffers = (
                "vision.attention.pos_encoder.pe",
                "language.token_encoder.pe",
                "language.pos_encoder.pe",
            )
            return key.startswith(vocabulary_modules) or key in positional_buffers

        backbone = lambda key: key.startswith("vision.backbone.")
        preserved = lambda key: not reset(key)
        return ModelBundle(
            name,
            model,
            source,
            reset,
            backbone,
            preserved,
            (32, 128),
            "abinet",
            [
                "vision.cls",
                "language.proj",
                "language.cls",
                "alignment.cls",
                "length-dependent positional buffers",
            ],
            compatibility_notes,
        )

    if name == "svtr":
        model = SVTRTransfer(dictionary_size, max_label_length)
        reset = lambda key: key.startswith("decoder.decoder.")
        backbone = lambda key: key.startswith("preprocessor.") or key.startswith("encoder.")
        return ModelBundle(
            name,
            model,
            source,
            reset,
            backbone,
            backbone,
            (64, 256),
            "ctc",
            ["decoder.decoder"],
            [
                "MMOCR STN/TPS rectification is forced to FP32 inside the "
                "outer AMP context; SVTR encoder and replacement CTC head "
                "remain AMP-enabled.",
            ],
        )

    raise ValueError(f"Unknown model: {name}")


def count_named_parameter_numel(model: nn.Module, keys: Iterable[str]) -> int:
    parameters = dict(model.named_parameters())
    return sum(parameters[key].numel() for key in keys if key in parameters)


def transfer_checkpoint_fail_closed(
    bundle: ModelBundle,
    minimum_backbone_loaded_ratio: float,
) -> Dict[str, object]:
    target_state = bundle.model.state_dict()
    parameter_names = set(dict(bundle.model.named_parameters()))
    loaded = OrderedDict()
    shape_mismatches = []
    unexpected = []
    forbidden_mismatches = []

    for key, value in bundle.source_state.items():
        target = target_state.get(key)
        if target is None:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target.shape):
            item = {
                "key": key,
                "checkpoint_shape": list(value.shape),
                "target_shape": list(target.shape),
                "allowed_reset": bundle.reset_key(key),
            }
            shape_mismatches.append(item)
            if not bundle.reset_key(key):
                forbidden_mismatches.append(item)
            continue
        if bundle.reset_key(key):
            continue
        loaded[key] = value

    missing = [key for key in target_state if key not in loaded]
    forbidden_missing = [key for key in missing if not bundle.reset_key(key)]
    forbidden_unexpected = [key for key in unexpected if not bundle.reset_key(key)]

    total_parameter_numel = sum(parameter.numel() for parameter in bundle.model.parameters())
    loaded_parameter_numel = count_named_parameter_numel(bundle.model, loaded.keys())
    backbone_parameters = [
        key for key in parameter_names if bundle.backbone_key(key)
    ]
    loaded_backbone_parameters = [key for key in loaded if bundle.backbone_key(key)]
    backbone_numel = count_named_parameter_numel(bundle.model, backbone_parameters)
    loaded_backbone_numel = count_named_parameter_numel(
        bundle.model, loaded_backbone_parameters
    )
    preserved_parameters = [
        key for key in parameter_names if bundle.preserved_body_key(key)
    ]
    loaded_preserved_parameters = [
        key for key in loaded if bundle.preserved_body_key(key)
    ]
    preserved_numel = count_named_parameter_numel(bundle.model, preserved_parameters)
    loaded_preserved_numel = count_named_parameter_numel(
        bundle.model, loaded_preserved_parameters
    )

    backbone_ratio = loaded_backbone_numel / backbone_numel if backbone_numel else 0.0
    preserved_ratio = loaded_preserved_numel / preserved_numel if preserved_numel else 0.0
    errors = []
    if forbidden_mismatches:
        errors.append("shape mismatch outside reset whitelist")
    if forbidden_missing:
        errors.append("target keys missing outside reset whitelist")
    if forbidden_unexpected:
        errors.append("checkpoint keys unexpected outside reset whitelist")
    if backbone_ratio + 1e-12 < minimum_backbone_loaded_ratio:
        errors.append(
            f"backbone loaded ratio {backbone_ratio:.12f} is below "
            f"{minimum_backbone_loaded_ratio:.12f}"
        )

    if errors:
        status = "failed"
    else:
        result = bundle.model.load_state_dict(loaded, strict=False)
        actual_missing = sorted(result.missing_keys)
        actual_unexpected = sorted(result.unexpected_keys)
        if actual_missing != sorted(missing) or actual_unexpected:
            errors.append("PyTorch load_state_dict result disagrees with audit")
            status = "failed"
        else:
            status = "passed"

    return {
        "status": status,
        "loaded_keys": len(loaded),
        "loaded_key_names": list(loaded),
        "missing_keys": missing,
        "shape_mismatch_keys": shape_mismatches,
        "unexpected_checkpoint_keys": unexpected,
        "loaded_parameter_ratio": loaded_parameter_numel / total_parameter_numel,
        "backbone_loaded_ratio": backbone_ratio,
        "preserved_body_loaded_ratio": preserved_ratio,
        "reset_modules": bundle.reset_modules,
        "parameter_counts": {
            "target_total": total_parameter_numel,
            "loaded": loaded_parameter_numel,
            "backbone_total": backbone_numel,
            "backbone_loaded": loaded_backbone_numel,
            "preserved_body_total": preserved_numel,
            "preserved_body_loaded": loaded_preserved_numel,
        },
        "errors": errors,
    }
