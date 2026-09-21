"""Dual-order labels for multilingual CTC and semantic guidance.

The LMDB label is a compact JSON payload produced by dual_order_protocol.py.
CTC and SMTR may independently consume visual or logical Unicode order while
the remaining fields describe the exact local bidi transport.
"""

from __future__ import annotations

import json
import unicodedata

import numpy as np

from openrec.preprocess.ctc_label_encode import CTCLabelEncode
from openrec.preprocess.smtr_label_encode import SMTRLabelEncode


SUPPORTED_LANGUAGES = {"zh", "ug", "kk"}
LOCAL_SCRIPT_IDS = {
    "han": 0,
    "arabic": 1,
    "cyrillic": 2,
    "common": 3,
}
ORDER_KEYS = {"visual": "visual_text", "logical": "logical_text"}
SUPPORTED_LABEL_VERSION = "DUAL_ORDER_LABEL_V2"


class DualOrderGTCLabelEncode:
    """Encode independent CTC/SGM orders plus order-transport metadata."""

    def __init__(
        self,
        gtc_label_encode,
        max_text_length,
        character_dict_path=None,
        use_space_char=False,
        ctc_order="visual",
        sgm_order="logical",
        **kwargs,
    ):
        if ctc_order not in ORDER_KEYS or sgm_order not in ORDER_KEYS:
            raise ValueError(
                f"Unsupported order pair: ctc={ctc_order}, sgm={sgm_order}"
            )
        if gtc_label_encode.get("name") != "SMTRLabelEncode":
            raise ValueError("DualOrderGTCLabelEncode requires SMTRLabelEncode")
        self.max_text_length = int(max_text_length)
        self.ctc_order = ctc_order
        self.sgm_order = sgm_order
        smtr_args = dict(gtc_label_encode)
        smtr_args.pop("name", None)
        self.sgm_encoder = SMTRLabelEncode(
            max_text_length=max_text_length,
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
            **smtr_args,
        )
        self.ctc_encoder = CTCLabelEncode(
            max_text_length=max_text_length,
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
        )

    @staticmethod
    def _parse_payload(raw_label):
        try:
            payload = json.loads(raw_label)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Dual-order LMDB labels must be JSON payloads"
            ) from exc
        required = {
            "version",
            "language",
            "logical_text",
            "visual_text",
            "logical_to_visual",
            "visual_directions",
            "removed_format_controls",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"Dual-order label is missing fields: {missing}")
        if payload["version"] != SUPPORTED_LABEL_VERSION:
            raise ValueError(
                f"Unsupported dual-order version: {payload['version']}"
            )
        return payload

    def _encode_tokens(self, text):
        encoded = self.ctc_encoder.encode(text)
        if encoded is None or len(encoded) != len(text):
            return None
        return encoded

    def _pad(self, values, fill):
        if len(values) > self.max_text_length:
            return None
        return np.asarray(
            list(values) + [fill] * (self.max_text_length - len(values)),
            dtype=np.int64,
        )

    @staticmethod
    def _local_script_id(char):
        codepoint = ord(char)
        name = unicodedata.name(char, "")
        if (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or "CJK UNIFIED IDEOGRAPH" in name
            or "CJK COMPATIBILITY IDEOGRAPH" in name
        ):
            return LOCAL_SCRIPT_IDS["han"]
        if (
            0x0600 <= codepoint <= 0x06FF
            or 0x0750 <= codepoint <= 0x077F
            or 0x08A0 <= codepoint <= 0x08FF
            or "ARABIC" in name
        ):
            return LOCAL_SCRIPT_IDS["arabic"]
        if (
            0x0400 <= codepoint <= 0x052F
            or 0x2DE0 <= codepoint <= 0x2DFF
            or 0xA640 <= codepoint <= 0xA69F
            or "CYRILLIC" in name
        ):
            return LOCAL_SCRIPT_IDS["cyrillic"]
        return LOCAL_SCRIPT_IDS["common"]

    def __call__(self, data):
        payload = self._parse_payload(data["label"])
        language = payload["language"]
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")

        logical = payload["logical_text"]
        visual = payload["visual_text"]
        if len(logical) != len(visual):
            raise ValueError("Logical and visual labels must have equal length")
        logical_to_visual = payload["logical_to_visual"]
        visual_directions = payload["visual_directions"]
        if sorted(logical_to_visual) != list(range(len(logical))):
            raise ValueError("logical_to_visual is not a valid permutation")
        if len(visual_directions) != len(visual):
            raise ValueError("visual_directions length mismatch")

        logical_ids = self._encode_tokens(logical)
        visual_ids = self._encode_tokens(visual)
        if logical_ids is None or visual_ids is None:
            return None

        ctc_text = payload[ORDER_KEYS[self.ctc_order]]
        sgm_text = payload[ORDER_KEYS[self.sgm_order]]
        ctc_data = self.ctc_encoder({"label": ctc_text})
        sgm_input = dict(data)
        sgm_input["label"] = sgm_text
        sgm_data = self.sgm_encoder(sgm_input)
        if ctc_data is None or sgm_data is None:
            return None

        permutation = self._pad(logical_to_visual, -1)
        visual_direction_ids = self._pad(visual_directions, -1)
        visual_script_ids = self._pad(
            [self._local_script_id(char) for char in visual], -1
        )
        logical_token_ids = self._pad(logical_ids, 0)
        visual_token_ids = self._pad(visual_ids, 0)
        if any(
            item is None
            for item in (
                permutation,
                visual_direction_ids,
                visual_script_ids,
                logical_token_ids,
                visual_token_ids,
            )
        ):
            return None

        consistency_mask = []
        for logical_index, visual_index in enumerate(logical_to_visual):
            consistency_mask.append(
                int(logical_ids[logical_index] == visual_ids[visual_index])
            )

        sgm_data["visual_token_ids"] = visual_token_ids
        sgm_data["logical_token_ids"] = logical_token_ids
        sgm_data["logical_to_visual"] = permutation
        sgm_data["visual_direction_ids"] = visual_direction_ids
        sgm_data["visual_script_ids"] = visual_script_ids
        sgm_data["consistency_mask"] = self._pad(consistency_mask, 0)
        sgm_data["ctc_label"] = ctc_data["label"]
        sgm_data["ctc_length"] = ctc_data["length"]
        return sgm_data
