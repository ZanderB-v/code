"""Dual-order semantic guidance decoder for multi-script recognition."""

from __future__ import annotations

import copy
import math
import unicodedata
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from openrec.modeling.decoders import build_decoder


def _unicode_script_id(character):
    """Map one dictionary entry to Han, Arabic, Cyrillic, or common."""

    if len(character) != 1:
        return 3
    codepoint = ord(character)
    name = unicodedata.name(character, "")
    if (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or "CJK UNIFIED IDEOGRAPH" in name
        or "CJK COMPATIBILITY IDEOGRAPH" in name
    ):
        return 0
    if (
        0x0600 <= codepoint <= 0x06FF
        or 0x0750 <= codepoint <= 0x077F
        or 0x08A0 <= codepoint <= 0x08FF
        or "ARABIC" in name
    ):
        return 1
    if (
        0x0400 <= codepoint <= 0x052F
        or 0x2DE0 <= codepoint <= 0x2DFF
        or 0xA640 <= codepoint <= 0xA69F
        or "CYRILLIC" in name
    ):
        return 2
    return 3


class ScriptAwareConfusionDiscriminator(nn.Module):
    """Training-only projection and same-script character prototype bank."""

    def __init__(
        self,
        channels,
        num_characters,
        character_dict_path,
        use_space_char=True,
    ):
        super().__init__()
        dictionary = Path(character_dict_path)
        if not dictionary.is_file():
            raise FileNotFoundError(
                f"SCDL character dictionary does not exist: {dictionary}"
            )
        characters = dictionary.read_text(encoding="utf-8-sig").splitlines()
        if use_space_char:
            characters.append(" ")
        if len(characters) != int(num_characters):
            raise ValueError(
                "SCDL dictionary/class mismatch: "
                f"{len(characters)} != {num_characters}"
            )

        self.projection = nn.Linear(channels, channels, bias=False)
        nn.init.eye_(self.projection.weight)
        self.register_buffer(
            "prototypes", torch.zeros(num_characters, channels)
        )
        self.register_buffer(
            "prototype_counts", torch.zeros(num_characters)
        )
        self.register_buffer(
            "prototype_script_ids",
            torch.tensor(
                [_unicode_script_id(character) for character in characters],
                dtype=torch.long,
            ),
        )

    def forward(self, frame_features):
        return F.normalize(self.projection(frame_features), dim=-1)


class _ScriptExpert(nn.Module):

    def __init__(self, channels, bottleneck_ratio=4):
        super().__init__()
        hidden = max(32, channels // bottleneck_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                hidden,
                hidden,
                3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x):
        return self.net(x)


class ScriptConditionedResidualAdapter(nn.Module):
    """Infer a local script field and softly mix residual visual experts."""

    def __init__(self, channels, num_scripts=4, bottleneck_ratio=4):
        super().__init__()
        hidden = max(32, channels // 4)
        self.script_head = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, num_scripts, 1),
        )
        self.experts = nn.ModuleList(
            [
                _ScriptExpert(channels, bottleneck_ratio)
                for _ in range(num_scripts)
            ]
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def predict_scripts(self, x):
        return self.script_head(x.mean(dim=2)).transpose(1, 2)

    def forward(self, x, script_logits=None):
        if script_logits is None:
            script_logits = self.predict_scripts(x)
        gates = F.softmax(script_logits, dim=-1).transpose(1, 2)
        expert_features = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )
        mixed = torch.einsum("bsw,bschw->bchw", gates, expert_features)
        scale = torch.tanh(self.residual_scale)
        return x + scale * mixed, script_logits


class ScriptAwareLocalDetailRefinement(nn.Module):
    """Lightweight 2-D detail refinement before the RCTC rearrangement.

    The module preserves visual order. It uses the M3 script posterior only to
    scale a shared visual residual; it does not reverse features or add a new
    language/ordering objective.
    """

    def __init__(
        self,
        channels,
        num_scripts=4,
        reduction=4,
        gamma_init=1e-3,
    ):
        super().__init__()
        if reduction <= 0:
            raise ValueError("SLDR reduction must be positive")
        if not math.isfinite(gamma_init) or gamma_init < 0:
            raise ValueError("SLDR gamma_init must be finite and non-negative")
        hidden = max(32, channels // reduction)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
        )
        self.local_3x3 = nn.Conv2d(
            hidden, hidden, 3, padding=1, groups=hidden, bias=False
        )
        self.local_1x5 = nn.Conv2d(
            hidden, hidden, (1, 5), padding=(0, 2), groups=hidden, bias=False
        )
        self.local_5x1 = nn.Conv2d(
            hidden, hidden, (5, 1), padding=(2, 0), groups=hidden, bias=False
        )
        gate_hidden = max(8, hidden // 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, gate_hidden, 1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, hidden, 1),
        )
        self.height_gate = nn.Conv2d(
            hidden,
            hidden,
            (3, 1),
            padding=(1, 0),
            groups=hidden,
        )
        self.width_gate = nn.Conv2d(
            hidden,
            hidden,
            (1, 3),
            padding=(0, 1),
            groups=hidden,
        )
        self.expand = nn.Conv2d(hidden, channels, 1, bias=False)
        self.script_scales = nn.Parameter(
            torch.full((num_scripts,), float(gamma_init))
        )

    def forward(self, x, script_logits):
        if x.dim() != 4:
            raise ValueError(f"SLDR expects BCHW features, got {tuple(x.shape)}")
        if script_logits.dim() != 3 or script_logits.shape[0] != x.shape[0]:
            raise ValueError("SLDR script logits must have shape [B, W, S]")
        if script_logits.shape[1] != x.shape[3]:
            raise ValueError(
                "SLDR script routing width differs from visual feature width: "
                f"{script_logits.shape[1]} != {x.shape[3]}"
            )
        if script_logits.shape[2] != self.script_scales.numel():
            raise ValueError("SLDR script count differs from script routing")

        reduced = self.reduce(x)
        local = (
            self.local_3x3(reduced)
            + self.local_1x5(reduced)
            + self.local_5x1(reduced)
        )
        channel = torch.sigmoid(self.channel_gate(local))
        height = torch.sigmoid(self.height_gate(local.mean(dim=3, keepdim=True)))
        width = torch.sigmoid(self.width_gate(local.mean(dim=2, keepdim=True)))
        attention = (channel + height + width) / 3.0
        residual = self.expand(local * (1.0 + attention))

        script_prob = F.softmax(script_logits.float(), dim=-1).to(x.dtype)
        column_scale = script_prob @ self.script_scales.to(x.dtype)
        return x + residual * column_scale[:, None, None, :]


class LocalDirectionConditioner(nn.Module):
    """Predict a local LTR/RTL field and modulate visual columns."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(32, channels // 4)
        self.direction_head = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, 2, 1),
        )
        self.scale_embed = nn.Parameter(torch.zeros(2, channels))
        self.bias_embed = nn.Parameter(torch.zeros(2, channels))
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        direction_logits = self.direction_head(x.mean(dim=2)).transpose(1, 2)
        probabilities = F.softmax(direction_logits, dim=-1)
        scale = torch.einsum("bwd,dc->bcw", probabilities, self.scale_embed)
        bias = torch.einsum("bwd,dc->bcw", probabilities, self.bias_embed)
        strength = torch.tanh(self.strength)
        conditioned = (
            x * (1.0 + strength * torch.tanh(scale).unsqueeze(2))
            + strength * bias.unsqueeze(2)
        )
        return conditioned, direction_logits


class DualOrderGTCDecoder(nn.Module):
    """RCTC visual branch plus logical SGM and optional condition adapters."""

    def __init__(
        self,
        in_channels,
        gtc_decoder,
        ctc_decoder,
        detach=False,
        infer_gtc=True,
        out_channels=0,
        use_script_adapter=False,
        use_local_direction=False,
        semantic_direction="none",
        direction_gate_initial=0.01,
        direction_gate_init_logit=-4.0,
        script_bottleneck_ratio=4,
        use_sldr=False,
        sldr_reduction=4,
        sldr_gamma_init=1e-3,
        use_scdl=False,
        scdl_character_dict_path=None,
        scdl_use_space_char=True,
        **kwargs,
    ):
        super().__init__()
        if not isinstance(out_channels, (list, tuple)) or len(out_channels) != 2:
            raise ValueError(
                "DualOrderGTCDecoder requires [SGM, CTC] output channels"
            )
        self.detach = detach
        self.infer_gtc = infer_gtc
        self.use_script_adapter = bool(use_script_adapter)
        self.use_local_direction = bool(use_local_direction)
        self.use_sldr = bool(use_sldr)
        self.use_scdl = bool(use_scdl)
        if self.use_sldr and not self.use_script_adapter:
            raise ValueError("SLDR requires the M3 script adapter for routing")
        if self.use_scdl and not self.use_script_adapter:
            raise ValueError("SCDL requires the complete M3 script adapter")
        if self.use_scdl and not scdl_character_dict_path:
            raise ValueError("SCDL requires scdl_character_dict_path")
        if semantic_direction not in ("none", "sgm_only", "script_gated", "sample_script_gated"):
            raise ValueError(f"Unknown semantic direction: {semantic_direction}")
        if self.use_local_direction and semantic_direction != "none":
            raise ValueError("CTC-side and SGM-side direction cannot be combined")
        if semantic_direction in ("script_gated", "sample_script_gated") and not self.use_script_adapter:
            raise ValueError("Script-gated direction requires the script adapter")
        self.semantic_direction = semantic_direction

        gtc_config = copy.deepcopy(gtc_decoder)
        ctc_config = copy.deepcopy(ctc_decoder)
        gtc_config["in_channels"] = in_channels
        gtc_config["out_channels"] = out_channels[0]
        ctc_config["in_channels"] = in_channels
        ctc_config["out_channels"] = out_channels[1]
        if self.use_scdl:
            ctc_config["return_feats"] = True
        self.gtc_decoder = build_decoder(gtc_config)
        self.ctc_decoder = build_decoder(ctc_config)

        self.script_adapter = (
            ScriptConditionedResidualAdapter(
                in_channels,
                num_scripts=4,
                bottleneck_ratio=script_bottleneck_ratio,
            )
            if self.use_script_adapter
            else None
        )
        self.sldr = (
            ScriptAwareLocalDetailRefinement(
                in_channels,
                num_scripts=4,
                reduction=sldr_reduction,
                gamma_init=sldr_gamma_init,
            )
            if self.use_sldr
            else None
        )
        self.scdl = (
            ScriptAwareConfusionDiscriminator(
                in_channels,
                num_characters=int(out_channels[1]) - 1,
                character_dict_path=scdl_character_dict_path,
                use_space_char=bool(scdl_use_space_char),
            )
            if self.use_scdl
            else None
        )
        self.direction_conditioner = (
            LocalDirectionConditioner(in_channels)
            if self.use_local_direction
            else None
        )
        self.semantic_direction_conditioner = (
            LocalDirectionConditioner(in_channels)
            if semantic_direction != "none" else None
        )
        if semantic_direction == "script_gated":
            if not 0 < direction_gate_initial < 1:
                raise ValueError("direction_gate_initial must be in (0, 1)")
            logit = math.log(direction_gate_initial / (1 - direction_gate_initial))
            self.direction_gate_logits = nn.Parameter(torch.full((4,), logit))
        elif semantic_direction == "sample_script_gated":
            if not math.isfinite(direction_gate_init_logit):
                raise ValueError("Gate logit must be finite")
            self.direction_gate_logits = nn.Parameter(torch.full((4,), float(direction_gate_init_logit)))

    def sample_script_gate(self, script_logits):
        # A single inferred dominant script per sample; no token-wise gate or GT language ID.
        script_id = F.softmax(script_logits.float(), dim=-1).mean(dim=1).argmax(dim=-1)
        return self.direction_gate_logits.sigmoid()[script_id]

    def forward(self, x, data=None):
        script_logits = None
        direction_logits = None
        shared = x
        if self.script_adapter is not None:
            if self.sldr is not None:
                # Reuse the frozen M3 routing head. Routing is inferred before
                # refinement so SLDR cannot manufacture its own script labels.
                script_logits = self.script_adapter.predict_scripts(shared)
                shared = self.sldr(shared, script_logits)
            shared, script_logits = self.script_adapter(
                shared, script_logits=script_logits
            )

        ctc_features = shared.detach() if self.detach else shared
        if self.semantic_direction != "none" and not self.detach:
            assert ctc_features is shared
            assert ctc_features.shape == shared.shape
        if self.direction_conditioner is not None:
            ctc_features, direction_logits = self.direction_conditioner(
                ctc_features
            )
        ctc_output = self.ctc_decoder(ctc_features, data=data)
        ctc_frame_features = None
        if self.use_scdl and self.training:
            if not isinstance(ctc_output, tuple) or len(ctc_output) != 2:
                raise ValueError("SCDL requires RCTCDecoder return_feats during training")
            ctc_frame_features, ctc_pred = ctc_output
        else:
            ctc_pred = ctc_output

        if self.training or self.infer_gtc:
            semantic = shared
            if self.semantic_direction_conditioner is not None:
                conditioned, direction_logits = self.semantic_direction_conditioner(shared)
                residual = conditioned - shared
                if self.semantic_direction == "script_gated":
                    # Routing uses the visual-column script posterior, never a language ID.
                    script_prob = F.softmax(script_logits, dim=-1)
                    gate = script_prob @ self.direction_gate_logits.sigmoid()
                    residual = residual * gate[:, None, None, :]
                elif self.semantic_direction == "sample_script_gated":
                    gate = self.sample_script_gate(script_logits)
                    residual = residual * gate[:, None, None, None]
                semantic = shared + residual
            gtc_features = semantic.flatten(2).transpose(1, 2)
            gtc_pred = self.gtc_decoder(gtc_features, data=data)
            output = {"gtc_pred": gtc_pred, "ctc_pred": ctc_pred}
            if script_logits is not None:
                output["script_logits"] = script_logits
            if direction_logits is not None:
                output["direction_logits"] = direction_logits
            if self.scdl is not None and self.training:
                output["scdl_frame_features"] = self.scdl(ctc_frame_features)
                output["scdl_prototypes"] = self.scdl.prototypes
                output["scdl_prototype_counts"] = self.scdl.prototype_counts
                output["scdl_prototype_script_ids"] = (
                    self.scdl.prototype_script_ids
                )
            return output
        return ctc_pred
