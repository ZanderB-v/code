"""Losses for dual-order multi-script semantic guidance."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from openrec.losses.ctc_loss import CTCLoss
from openrec.losses.smtr_loss import SMTRLoss


class DualOrderGTCLoss(nn.Module):
    """CTC + logical SGM + bidi posterior transport consistency."""

    def __init__(
        self,
        gtc_weight=1.0,
        ctc_weight=0.1,
        consistency_weight=0.0,
        script_weight=0.0,
        direction_weight=0.0,
        consistency_temperature=1.0,
        alignment_prior_scale=0.75,
        scdl_weight=0.0,
        scdl_topk=5,
        scdl_temperature=0.1,
        scdl_warmup_fraction=0.2,
        zero_infinity=True,
        gtc_loss=None,
        **kwargs,
    ):
        super().__init__()
        self.ctc_loss = CTCLoss(zero_infinity=zero_infinity)
        if gtc_loss is not None and gtc_loss.get("name") != "SMTRLoss":
            raise ValueError("DualOrderGTCLoss requires the standard SMTRLoss")
        self.gtc_loss = SMTRLoss()
        self.gtc_weight = float(gtc_weight)
        self.ctc_weight = float(ctc_weight)
        self.consistency_weight = float(consistency_weight)
        self.script_weight = float(script_weight)
        self.direction_weight = float(direction_weight)
        self.temperature = float(consistency_temperature)
        self.alignment_prior_scale = float(alignment_prior_scale)
        self.scdl_weight = float(scdl_weight)
        self.scdl_topk = int(scdl_topk)
        self.scdl_temperature = float(scdl_temperature)
        self.scdl_warmup_fraction = float(scdl_warmup_fraction)
        self._training_step = 0
        self._training_total_steps = 1
        if self.scdl_weight < 0:
            raise ValueError("scdl_weight must be non-negative")
        if self.scdl_topk < 1:
            raise ValueError("scdl_topk must be positive")
        if self.scdl_temperature <= 0:
            raise ValueError("scdl_temperature must be positive")
        if not 0 <= self.scdl_warmup_fraction < 1:
            raise ValueError("scdl_warmup_fraction must be in [0, 1)")

    def set_training_progress(self, global_step, total_steps):
        self._training_step = max(0, int(global_step))
        self._training_total_steps = max(1, int(total_steps))

    def _scdl_active_weight(self):
        progress = self._training_step / self._training_total_steps
        return self.scdl_weight if progress >= self.scdl_warmup_fraction else 0.0

    @staticmethod
    def _normalized_character_probabilities(
        logits, temperature, drop_last=False
    ):
        end = -1 if drop_last else None
        probabilities = F.softmax(logits / temperature, dim=-1)[..., 1:end]
        return probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-7)

    @staticmethod
    def _ctc_character_distributions(logits, temperature):
        """Return absolute character mass, conditional characters, and mass.

        CTC blank must remain visible while estimating token-to-frame
        alignment. Renormalizing after dropping blank would otherwise make a
        nearly blank frame look like confident character evidence.
        """

        probabilities = F.softmax(logits / temperature, dim=-1)
        character_mass = probabilities[..., 1:]
        nonblank_mass = character_mass.sum(dim=-1)
        conditional = character_mass / nonblank_mass.unsqueeze(-1).clamp_min(
            1e-7
        )
        return character_mass, conditional, nonblank_mass

    @staticmethod
    def _inverse_entropy_confidence(probabilities):
        class_count = probabilities.shape[-1]
        entropy_scale = probabilities.new_tensor(float(class_count)).log()
        entropy_scale = entropy_scale.clamp_min(1e-7)
        entropy = -(
            probabilities.clamp_min(1e-7)
            * probabilities.clamp_min(1e-7).log()
        ).sum(dim=-1)
        return (1.0 - entropy / entropy_scale).clamp(0.0, 1.0)

    def _token_frame_scores(self, sample_ctc, visual_target):
        """Score visual tokens against frames with a soft monotonic prior."""

        frame_count = sample_ctc.shape[0]
        length = visual_target.shape[0]
        token_positions = torch.arange(
            length, device=sample_ctc.device, dtype=sample_ctc.dtype
        )
        frame_positions = torch.arange(
            frame_count,
            device=sample_ctc.device,
            dtype=sample_ctc.dtype,
        )
        centers = (
            (token_positions + 0.5) * frame_count / max(length, 1) - 0.5
        )
        sigma = max(
            1.0,
            frame_count / max(length, 1) * self.alignment_prior_scale,
        )
        monotonic_prior = -0.5 * (
            (frame_positions.unsqueeze(0) - centers.unsqueeze(1)) / sigma
        ).square()
        target_likelihood = sample_ctc[
            :, visual_target.clamp_min(0)
        ].transpose(0, 1)
        return target_likelihood.clamp_min(1e-7).log() + monotonic_prior

    def _cross_order_consistency(self, predicts, batch):
        ctc_logits = predicts["ctc_pred"]
        semantic_logits = predicts["gtc_pred"][1]
        lengths = batch[8].long()
        visual_ids = batch[9].long()
        logical_ids = batch[10].long()
        logical_to_visual = batch[11].long()
        consistency_mask = batch[14].bool()

        ctc_mass, ctc_char, _ = self._ctc_character_distributions(
            ctc_logits, self.temperature
        )
        sgm_char = self._normalized_character_probabilities(
            semantic_logits,
            self.temperature,
            drop_last=True,
        )
        if ctc_char.shape[-1] != sgm_char.shape[-1]:
            raise ValueError(
                "CTC and SGM character spaces differ after removing specials: "
                f"{ctc_char.shape[-1]} != {sgm_char.shape[-1]}"
            )
        losses = []
        for batch_index, raw_length in enumerate(lengths):
            length = int(raw_length.item())
            if length <= 0:
                continue
            visual_target = visual_ids[batch_index, :length] - 1
            logical_target = logical_ids[batch_index, :length] - 1
            permutation = logical_to_visual[batch_index, :length]
            transported_target = visual_target[permutation.clamp_min(0)]
            valid = (
                consistency_mask[batch_index, :length]
                & (visual_target >= 0)
                & (logical_target >= 0)
                & (permutation >= 0)
                & (transported_target == logical_target)
            )
            if not valid.any():
                continue

            sample_ctc_mass = ctc_mass[batch_index]
            sample_ctc = ctc_char[batch_index]
            token_frame_scores = self._token_frame_scores(
                sample_ctc_mass, visual_target
            )
            alignment = F.softmax(token_frame_scores, dim=-1)
            visual_token_posteriors = alignment @ sample_ctc
            logical_ctc = visual_token_posteriors[permutation.clamp_min(0)]
            logical_sgm = sgm_char[batch_index, :length]

            p = logical_ctc[valid].clamp_min(1e-7)
            q = logical_sgm[valid].clamp_min(1e-7)
            p = p / p.sum(dim=-1, keepdim=True)
            q = q / q.sum(dim=-1, keepdim=True)
            midpoint = 0.5 * (p + q)
            js_per_token = 0.5 * (
                (p * (p.log() - midpoint.log())).sum(dim=-1)
                + (q * (q.log() - midpoint.log())).sum(dim=-1)
            )
            p_confidence = self._inverse_entropy_confidence(p)
            q_confidence = self._inverse_entropy_confidence(q)
            reliability = torch.sqrt(
                p_confidence.clamp(0.0, 1.0)
                * q_confidence.clamp(0.0, 1.0)
            ).detach()
            losses.append(
                (js_per_token * reliability).sum()
                / reliability.sum().clamp_min(1e-7)
            )
        if not losses:
            return ctc_logits.sum() * 0.0
        return torch.stack(losses).mean()

    def _direction_loss(self, direction_logits, ctc_logits, batch):
        lengths = batch[8].long()
        visual_ids = batch[9].long()
        visual_directions = batch[12].long()
        ctc_mass, _, nonblank_mass = self._ctc_character_distributions(
            ctc_logits, self.temperature
        )
        losses = []
        for batch_index, raw_length in enumerate(lengths):
            length = int(raw_length.item())
            if length <= 0:
                continue
            visual_target = visual_ids[batch_index, :length] - 1
            directions = visual_directions[batch_index, :length]
            valid_tokens = (visual_target >= 0) & (directions >= 0)
            if not valid_tokens.any():
                continue

            token_frame_scores = self._token_frame_scores(
                ctc_mass[batch_index], visual_target
            )
            token_frame_scores = token_frame_scores.masked_fill(
                ~valid_tokens.unsqueeze(1), float("-inf")
            )
            frame_token_weights = F.softmax(
                token_frame_scores, dim=0
            ).detach()
            direction_targets = F.one_hot(
                directions.clamp(0, 1), num_classes=2
            ).to(frame_token_weights.dtype)
            frame_targets = (
                frame_token_weights.transpose(0, 1) @ direction_targets
            )
            frame_targets = frame_targets / frame_targets.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-7)
            frame_log_prob = F.log_softmax(
                direction_logits[batch_index], dim=-1
            )
            frame_loss = -(frame_targets * frame_log_prob).sum(dim=-1)
            frame_weight = nonblank_mass[batch_index].detach()
            losses.append(
                (frame_loss * frame_weight).sum()
                / frame_weight.sum().clamp_min(1e-7)
            )
        if not losses:
            return direction_logits.sum() * 0.0
        return torch.stack(losses).mean()

    def _script_loss(self, script_logits, ctc_logits, batch):
        lengths = batch[8].long()
        visual_ids = batch[9].long()
        visual_scripts = batch[13].long()
        ctc_mass, _, nonblank_mass = self._ctc_character_distributions(
            ctc_logits, self.temperature
        )
        num_scripts = script_logits.shape[-1]
        losses = []
        for batch_index, raw_length in enumerate(lengths):
            length = int(raw_length.item())
            if length <= 0:
                continue
            visual_target = visual_ids[batch_index, :length] - 1
            scripts = visual_scripts[batch_index, :length]
            valid_tokens = (visual_target >= 0) & (scripts >= 0)
            if not valid_tokens.any():
                continue

            token_frame_scores = self._token_frame_scores(
                ctc_mass[batch_index], visual_target
            )
            token_frame_scores = token_frame_scores.masked_fill(
                ~valid_tokens.unsqueeze(1), float("-inf")
            )
            frame_token_weights = F.softmax(
                token_frame_scores, dim=0
            ).detach()
            script_targets = F.one_hot(
                scripts.clamp(0, num_scripts - 1),
                num_classes=num_scripts,
            ).to(frame_token_weights.dtype)
            frame_targets = (
                frame_token_weights.transpose(0, 1) @ script_targets
            )
            frame_targets = frame_targets / frame_targets.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-7)
            frame_log_prob = F.log_softmax(
                script_logits[batch_index], dim=-1
            )
            frame_loss = -(frame_targets * frame_log_prob).sum(dim=-1)
            frame_weight = nonblank_mass[batch_index].detach()
            losses.append(
                (frame_loss * frame_weight).sum()
                / frame_weight.sum().clamp_min(1e-7)
            )
        if not losses:
            return script_logits.sum() * 0.0
        return torch.stack(losses).mean()

    @staticmethod
    def _ctc_token_posteriors(ctc_logits, visual_ids, lengths):
        """Exact batched CTC token occupancy for unambiguous target positions.

        Each target position is represented as its own temporary CTC class.
        This is exactly equivalent to the original CTC lattice when adjacent
        target characters differ. Samples with adjacent repeats are excluded:
        assigning distinct position classes would incorrectly permit a direct
        repeated-character transition, while sharing a class would lose token
        identity. This conservative exclusion avoids approximate supervision.
        """

        log_prob = F.log_softmax(ctc_logits.detach().float(), dim=-1)
        batch_size, frame_count, _ = log_prob.shape
        max_tokens = int(lengths.max().item()) if lengths.numel() else 0
        empty = log_prob.new_zeros((batch_size, max_tokens, frame_count))
        if max_tokens <= 0:
            return empty, torch.zeros(
                batch_size, dtype=torch.bool, device=log_prob.device
            )

        token_index = torch.arange(max_tokens, device=log_prob.device)[None, :]
        token_valid = token_index < lengths[:, None]
        adjacent_repeat = (
            (visual_ids[:, 1:max_tokens] == visual_ids[:, : max_tokens - 1])
            & token_valid[:, 1:]
        ).any(dim=1)
        valid_sample = (
            (lengths > 0)
            & (lengths <= frame_count)
            & ~adjacent_repeat
        )
        selected = valid_sample.nonzero(as_tuple=False).squeeze(1)
        if selected.numel() == 0:
            return empty, valid_sample

        selected_lengths = lengths[selected].long()
        selected_targets = visual_ids[selected, :max_tokens].long()
        selected_log_prob = log_prob[selected]
        token_emission = selected_log_prob.gather(
            2,
            selected_targets[:, None, :].expand(
                selected.numel(), frame_count, max_tokens
            ),
        )
        raw_emission = torch.cat(
            (selected_log_prob[:, :, :1], token_emission), dim=-1
        )

        # Frame-wise normalization changes every valid path by the same factor,
        # so it leaves CTC posteriors unchanged while enabling the stable native
        # CTCLoss backward formula: posterior = probability - gradient.
        with torch.enable_grad():
            positional_log_prob = raw_emission.log_softmax(dim=-1)
            positional_log_prob = positional_log_prob.detach().requires_grad_(True)
            position_targets = torch.cat(
                [
                    torch.arange(
                        1,
                        int(length.item()) + 1,
                        device=log_prob.device,
                        dtype=torch.long,
                    )
                    for length in selected_lengths
                ]
            )
            input_lengths = torch.full_like(selected_lengths, frame_count)
            nll = F.ctc_loss(
                positional_log_prob.transpose(0, 1),
                position_targets,
                input_lengths,
                selected_lengths,
                blank=0,
                reduction="sum",
                zero_infinity=True,
            )
            gradient = torch.autograd.grad(
                nll, positional_log_prob, create_graph=False
            )[0]
            occupancy = (positional_log_prob.exp() - gradient).clamp_min(0.0)

        gamma = empty
        gamma[selected] = occupancy[:, :, 1:].permute(0, 2, 1).detach()
        return gamma, valid_sample

    @staticmethod
    @torch.no_grad()
    def _update_scdl_prototypes(prototypes, counts, features, labels):
        class_sums = torch.zeros_like(prototypes)
        class_counts = torch.zeros_like(counts)
        class_sums.index_add_(0, labels, features)
        class_counts.index_add_(
            0, labels, torch.ones_like(labels, dtype=counts.dtype)
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(class_sums)
            torch.distributed.all_reduce(class_counts)
        observed = class_counts > 0
        if not observed.any():
            return
        total = counts[observed] + class_counts[observed]
        updated = (
            prototypes[observed] * counts[observed, None]
            + class_sums[observed]
        ) / total[:, None].clamp_min(1.0)
        prototypes[observed] = F.normalize(updated, dim=-1)
        counts[observed] = total

    def _scdl_loss(self, predicts, batch):
        required = (
            "scdl_frame_features",
            "scdl_prototypes",
            "scdl_prototype_counts",
            "scdl_prototype_script_ids",
        )
        missing = [name for name in required if name not in predicts]
        if missing:
            raise ValueError(f"SCDL outputs are missing: {missing}")

        frame_features = predicts["scdl_frame_features"]
        prototypes = predicts["scdl_prototypes"]
        counts = predicts["scdl_prototype_counts"]
        prototype_scripts = predicts["scdl_prototype_script_ids"]
        lengths = batch[8].long()
        visual_ids = batch[9].long()
        visual_scripts = batch[13].long()
        gamma, valid_samples = self._ctc_token_posteriors(
            predicts["ctc_pred"], visual_ids, lengths
        )
        pooled = gamma @ frame_features
        pooled = pooled / gamma.sum(dim=-1, keepdim=True).clamp_min(1e-7)
        pooled = F.normalize(pooled, dim=-1)

        max_tokens = pooled.shape[1]
        token_index = torch.arange(max_tokens, device=pooled.device)[None, :]
        valid = (
            (token_index < lengths[:, None])
            & valid_samples[:, None]
            & (visual_ids[:, :max_tokens] > 0)
            & (visual_scripts[:, :max_tokens] >= 0)
            & (visual_scripts[:, :max_tokens] < 3)
        )
        token_features = pooled[valid]
        token_labels = visual_ids[:, :max_tokens][valid] - 1
        token_scripts = visual_scripts[:, :max_tokens][valid]
        zero = frame_features.sum() * 0.0
        if token_features.numel() == 0:
            return (
                zero,
                0,
                int(valid.sum().item()),
                int(valid_samples.sum().item()),
            )

        bank = prototypes.detach().clone()
        bank_counts = counts.detach().clone()
        script_losses = []
        valid_token_count = 0
        for script_id in range(3):
            script_mask = token_scripts == script_id
            candidate_mask = (prototype_scripts == script_id) & (bank_counts > 0)
            if not script_mask.any() or candidate_mask.sum() < 2:
                script_losses.append(zero)
                continue
            features = token_features[script_mask]
            labels = token_labels[script_mask]
            positive_ready = bank_counts[labels] > 0
            features = features[positive_ready]
            labels = labels[positive_ready]
            if features.numel() == 0:
                script_losses.append(zero)
                continue
            candidate_ids = candidate_mask.nonzero(as_tuple=False).squeeze(1)
            candidate_prototypes = bank[candidate_ids]
            similarity = features @ candidate_prototypes.transpose(0, 1)
            same_class = labels[:, None] == candidate_ids[None, :]
            negatives = similarity.masked_fill(same_class, float("-inf"))
            k = min(self.scdl_topk, max(1, candidate_ids.numel() - 1))
            hard_negative = negatives.topk(k, dim=1).values
            positive = (features * bank[labels]).sum(dim=-1, keepdim=True)
            logits = torch.cat((positive, hard_negative), dim=1)
            targets = torch.zeros(
                logits.shape[0], dtype=torch.long, device=logits.device
            )
            script_losses.append(
                F.cross_entropy(logits / self.scdl_temperature, targets)
            )
            valid_token_count += int(logits.shape[0])

        loss = torch.stack(script_losses).sum() / 3.0
        self._update_scdl_prototypes(
            prototypes,
            counts,
            token_features.detach(),
            token_labels.detach(),
        )
        return (
            loss,
            valid_token_count,
            int(valid.sum().item()),
            int(valid_samples.sum().item()),
        )

    def forward(self, predicts, batch):
        ctc_loss = self.ctc_loss(
            predicts["ctc_pred"], [None] + batch[-2:]
        )["loss"]
        gtc_loss = self.gtc_loss(predicts["gtc_pred"], batch[:-2])["loss"]
        zero = ctc_loss.detach() * 0.0
        consistency_loss = zero
        script_loss = zero
        direction_loss = zero
        scdl_loss = zero
        scdl_valid_tokens = 0
        scdl_aligned_tokens = 0
        scdl_valid_samples = 0

        if self.consistency_weight > 0:
            consistency_loss = self._cross_order_consistency(predicts, batch)
        if self.script_weight > 0:
            script_logits = predicts.get("script_logits")
            if script_logits is None:
                raise ValueError("script_weight > 0 but adapter is disabled")
            script_loss = self._script_loss(
                script_logits, predicts["ctc_pred"], batch
            )
        if self.direction_weight > 0:
            direction_logits = predicts.get("direction_logits")
            if direction_logits is None:
                raise ValueError(
                    "direction_weight > 0 but local direction is disabled"
                )
            direction_loss = self._direction_loss(
                direction_logits, predicts["ctc_pred"], batch
            )
        active_scdl_weight = self._scdl_active_weight()
        # Warmup disables both the discriminative gradient and prototype-bank
        # updates. Early CTC alignments must not seed persistent prototypes.
        if active_scdl_weight > 0:
            (
                scdl_loss,
                scdl_valid_tokens,
                scdl_aligned_tokens,
                scdl_valid_samples,
            ) = self._scdl_loss(predicts, batch)

        total = (
            self.ctc_weight * ctc_loss
            + self.gtc_weight * gtc_loss
            + self.consistency_weight * consistency_loss
            + self.script_weight * script_loss
            + self.direction_weight * direction_loss
            + active_scdl_weight * scdl_loss
        )
        return {
            "loss": total,
            "ctc_loss": ctc_loss,
            "gtc_loss": gtc_loss,
            "cross_order_loss": consistency_loss,
            "script_loss": script_loss,
            "direction_loss": direction_loss,
            "scdl_loss": scdl_loss,
            "scdl_active_weight": ctc_loss.new_tensor(active_scdl_weight),
            "scdl_valid_tokens": ctc_loss.new_tensor(scdl_valid_tokens),
            "scdl_aligned_tokens": ctc_loss.new_tensor(scdl_aligned_tokens),
            "scdl_valid_samples": ctc_loss.new_tensor(scdl_valid_samples),
            "scdl_excluded_alignment_samples": ctc_loss.new_tensor(
                int(predicts["ctc_pred"].shape[0]) - scdl_valid_samples
                if self.scdl_weight > 0
                else 0
            ),
        }
