import math
import os
import random
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


SAMPLER_PROTOCOL = 'RATIO_SAMPLER_DDP_BALANCED_V2'
HEM_PROTOCOL = 'NEAR_MISS_HEM_WEIGHTED_RATIO_V1'


def partition_indices_by_ratio(wh_ratio, num_replicas, rank):
    """Assign an equal number of samples per ratio bucket to every rank."""
    if num_replicas < 1:
        raise ValueError(f'num_replicas must be positive: {num_replicas}')
    if rank < 0 or rank >= num_replicas:
        raise ValueError(f'invalid rank {rank}/{num_replicas}')

    assigned = []
    padding_count = 0
    for ratio in np.unique(wh_ratio):
        ratio_ids = np.flatnonzero(wh_ratio == ratio)
        if ratio_ids.size == 0:
            continue
        target_count = int(math.ceil(ratio_ids.size / num_replicas))
        rank_ids = ratio_ids[rank::num_replicas]
        missing = target_count - rank_ids.size
        if missing > 0:
            # Padding is deterministic and confined to the same ratio bucket.
            # It guarantees identical batch shapes and step counts on all DDP
            # ranks while retaining every original sample at least once.
            offsets = (np.arange(missing) + rank) % ratio_ids.size
            rank_ids = np.concatenate([rank_ids, ratio_ids[offsets]])
            padding_count += missing
        assigned.append(rank_ids)

    if not assigned:
        return np.asarray([], dtype=np.int64), padding_count
    return np.concatenate(assigned).astype(np.int64), padding_count


def merge_control_world_indices_by_ratio(wh_ratio, control_world_size):
    """Merge the exact per-rank sample multisets of a DDP control run."""
    assigned = []
    padding_count = 0
    for ratio in np.unique(wh_ratio):
        ratio_ids = np.flatnonzero(wh_ratio == ratio)
        if ratio_ids.size == 0:
            continue
        target_count = int(math.ceil(ratio_ids.size / control_world_size))
        rank_parts = []
        for rank in range(control_world_size):
            rank_ids = ratio_ids[rank::control_world_size]
            missing = target_count - rank_ids.size
            if missing > 0:
                offsets = (np.arange(missing) + rank) % ratio_ids.size
                rank_ids = np.concatenate([rank_ids, ratio_ids[offsets]])
                padding_count += missing
            rank_parts.append(rank_ids)
        assigned.append(np.concatenate(rank_parts))
    if not assigned:
        return np.asarray([], dtype=np.int64), padding_count
    return np.concatenate(assigned).astype(np.int64), padding_count


class RatioSampler(Sampler):

    def __init__(self,
                 data_source,
                 scales,
                 first_bs=512,
                 fix_bs=True,
                 divided_factor=[8, 16],
                 is_training=True,
                 max_ratio=10,
                 max_bs=1024,
                 seed=None,
                 sample_weight_manifest=None,
                 hem_protocol=None,
                 hem_seed=20260731,
                 control_world_size=1,
                 control_first_bs=None):
        """
            multi scale samper
            Args:
                data_source(dataset)
                scales(list): several scales for image resolution
                first_bs(int): batch size for the first scale in scales
                divided_factor(list[w, h]): ImageNet models down-sample images by a factor, ensure that width and height dimensions are multiples are multiple of devided_factor.
                is_training(boolean): mode
        """
        # min. and max. spatial dimensions
        self.data_source = data_source
        # self.data_idx_order_list = np.array(data_source.data_idx_order_list)
        self.ds_width = data_source.ds_width
        self.seed = data_source.seed
        if self.ds_width:
            self.wh_ratio = data_source.wh_ratio
            self.wh_ratio_sort = data_source.wh_ratio_sort
        self.n_data_samples = len(self.data_source)
        self.max_ratio = max_ratio
        self.max_bs = max_bs

        if isinstance(scales[0], list):
            width_dims = [i[0] for i in scales]
            height_dims = [i[1] for i in scales]
        elif isinstance(scales[0], int):
            width_dims = scales
            height_dims = scales
        base_im_w = width_dims[0]
        base_im_h = height_dims[0]
        base_batch_size = first_bs
        base_elements = base_im_w * base_im_h * base_batch_size
        self.base_elements = base_elements
        self.base_batch_size = base_batch_size
        self.base_im_h = base_im_h
        self.base_im_w = base_im_w

        # Use the active process group, not the number of visible devices.
        # A single-process run may deliberately expose multiple GPUs; treating
        # every visible GPU as a replica silently drops part of the dataset.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            num_replicas = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            num_replicas = 1
            rank = 0
        # self.rank = rank
        img_indices = [idx for idx in range(self.n_data_samples)]
        self.shuffle = False
        if is_training:
            # compute the spatial dimensions and corresponding batch size
            # ImageNet models down-sample images by a factor of 32.
            # Ensure that width and height dimensions are multiples are multiple of 32.
            width_dims = [
                int((w // divided_factor[0]) * divided_factor[0])
                for w in width_dims
            ]
            height_dims = [
                int((h // divided_factor[1]) * divided_factor[1])
                for h in height_dims
            ]

            img_batch_pairs = list()
            for (h, w) in zip(height_dims, width_dims):
                if fix_bs:
                    batch_size = base_batch_size
                else:
                    batch_size = int(max(1, (base_elements / (h * w))))
                img_batch_pairs.append((w, h, batch_size))
            self.img_batch_pairs = img_batch_pairs
            self.shuffle = True
            np.random.seed(seed)
            random.seed(seed)
        else:
            self.img_batch_pairs = [(base_im_w, base_im_h, base_batch_size)]

        self.img_indices = img_indices
        self.epoch = 0
        self.rank = rank
        self.num_replicas = num_replicas
        self.sampler_protocol = SAMPLER_PROTOCOL
        self.hem_protocol = None
        self.sample_weight_manifest = None
        self.sample_weight_manifest_sha256 = None
        self.dataset_sample_weights = None
        self.hem_seed = int(hem_seed)
        self.control_world_size = int(control_world_size)
        self.control_first_bs = int(
            control_first_bs if control_first_bs is not None else first_bs)
        if self.control_world_size < 1 or self.control_first_bs < 1:
            raise ValueError('HEM control world size and batch size must be positive')
        if self.num_replicas == 1 and self.control_world_size > 1:
            expected_batch_size = self.control_first_bs * self.control_world_size
            if self.base_batch_size != expected_batch_size:
                raise ValueError(
                    'Single-process control emulation requires first_bs='
                    f'{expected_batch_size}, got {self.base_batch_size}')
        elif self.num_replicas > 1 and self.control_world_size > 1:
            if self.num_replicas != self.control_world_size:
                raise ValueError(
                    'Actual and control DDP world sizes differ: '
                    f'{self.num_replicas} != {self.control_world_size}')
            if self.base_batch_size != self.control_first_bs:
                raise ValueError(
                    'DDP control run requires its original per-rank batch: '
                    f'{self.control_first_bs}, got {self.base_batch_size}')
        if sample_weight_manifest is not None:
            if hem_protocol != HEM_PROTOCOL:
                raise ValueError(
                    f'HEM protocol mismatch: {hem_protocol!r} != {HEM_PROTOCOL!r}')
            self._load_sample_weights(sample_weight_manifest)
            if self.control_world_size != 1 and self.num_replicas != 1:
                raise ValueError(
                    'HEM control-world emulation requires one actual process')

        # self.batch_list = []
        self.current = 0
        self.is_training = is_training
        if is_training and self.dataset_sample_weights is not None:
            indices_rank_i_ori, distributed_padding = self._weighted_rank_indices(0)
        elif is_training and num_replicas == 1 and self.control_world_size > 1:
            indices_rank_i_ori, distributed_padding = (
                merge_control_world_indices_by_ratio(
                    self.wh_ratio,
                    self.control_world_size,
                )
            )
        elif is_training:
            indices_rank_i_ori, distributed_padding = partition_indices_by_ratio(
                self.wh_ratio,
                self.num_replicas,
                self.rank,
            )
        else:
            indices_rank_i_ori = np.asarray(
                self.wh_ratio_sort[self.img_indices], dtype=np.int64)
            distributed_padding = 0
        self.indices_rank_i_ori = indices_rank_i_ori
        self.indices_rank_i_ratio = self.wh_ratio[self.indices_rank_i_ori]
        self.distributed_padding = distributed_padding
        self.n_samples_per_replica = len(self.indices_rank_i_ori)
        indices_rank_i_ratio_unique = np.unique(self.indices_rank_i_ratio)
        self.indices_rank_i_ratio_unique = indices_rank_i_ratio_unique.tolist()
        self.batch_list = self.create_batch()
        self.length = len(self.batch_list)
        self.batchs_in_one_epoch_id = [i for i in range(self.length)]

    def _load_sample_weights(self, manifest_path):
        path = Path(manifest_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        records = []
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for raw_line in handle:
                digest.update(raw_line)
                if raw_line.strip():
                    records.append(json.loads(raw_line.decode('utf-8')))
        if len(records) != self.n_data_samples:
            raise ValueError(
                f'HEM manifest rows {len(records)} != dataset rows {self.n_data_samples}')
        if len(self.data_source.lmdb_sets) != 1:
            raise ValueError('HEM V1 supports exactly one frozen target-train LMDB')
        manifest_weights = np.asarray(
            [float(record['sample_weight']) for record in records], dtype=np.float64)
        allowed = {1.0, 1.5, 2.0}
        if set(np.unique(manifest_weights).tolist()) - allowed:
            raise ValueError('HEM manifest contains an unregistered sample weight')
        dataset_weights = np.empty(self.n_data_samples, dtype=np.float64)
        for dataset_index, (lmdb_index, file_index) in enumerate(
                self.data_source.data_idx_order_list):
            if int(lmdb_index) != 0:
                raise ValueError('HEM V1 encountered an unexpected LMDB index')
            file_index = int(file_index)
            if file_index < 1 or file_index > len(manifest_weights):
                raise ValueError(f'HEM LMDB file index is out of range: {file_index}')
            dataset_weights[dataset_index] = manifest_weights[file_index - 1]
        self.dataset_sample_weights = dataset_weights
        self.sample_weight_manifest = str(path)
        self.sample_weight_manifest_sha256 = digest.hexdigest()
        self.hem_protocol = HEM_PROTOCOL

    def _weighted_rank_indices(self, epoch):
        assigned = []
        padding_count = 0
        for ratio in np.unique(self.wh_ratio):
            ratio_ids = np.flatnonzero(self.wh_ratio == ratio)
            if ratio_ids.size == 0:
                continue
            sampling_replicas = (
                self.control_world_size
                if self.num_replicas == 1
                else self.num_replicas
            )
            target_count = int(math.ceil(ratio_ids.size / sampling_replicas))
            draw_count = target_count * sampling_replicas
            padding_count += draw_count - ratio_ids.size
            weights = self.dataset_sample_weights[ratio_ids]
            probabilities = weights / weights.sum()
            ratio_seed = (
                self.hem_seed * 1000003
                + int(epoch) * 9176
                + int(ratio) * 131
            ) % (2**63 - 1)
            rng = np.random.default_rng(ratio_seed)
            global_draws = rng.choice(
                ratio_ids,
                size=draw_count,
                replace=True,
                p=probabilities,
            )
            if self.num_replicas == 1:
                assigned.append(global_draws)
            else:
                assigned.append(global_draws[self.rank::self.num_replicas])
        if not assigned:
            return np.asarray([], dtype=np.int64), padding_count
        return np.concatenate(assigned).astype(np.int64), padding_count

    def hem_selection_summary(self):
        if self.dataset_sample_weights is None:
            return None
        selected_weights = self.dataset_sample_weights[self.indices_rank_i_ori]
        values, counts = np.unique(selected_weights, return_counts=True)
        population_values, population_counts = np.unique(
            self.dataset_sample_weights, return_counts=True)
        return {
            'protocol': self.hem_protocol,
            'manifest': self.sample_weight_manifest,
            'manifest_sha256': self.sample_weight_manifest_sha256,
            'epoch': self.epoch,
            'actual_world_size': self.num_replicas,
            'control_world_size': self.control_world_size,
            'control_first_bs': self.control_first_bs,
            'selected_weight_counts': {
                str(float(value)): int(count)
                for value, count in zip(values, counts)
            },
            'population_weight_counts': {
                str(float(value)): int(count)
                for value, count in zip(population_values, population_counts)
            },
        }

    def create_batch(self):
        batch_list = []
        for ratio in self.indices_rank_i_ratio_unique:
            ratio_ids = np.where(self.indices_rank_i_ratio == ratio)[0]
            ratio_ids = self.indices_rank_i_ori[ratio_ids]
            if self.shuffle:
                random.shuffle(ratio_ids)
            num_ratio = ratio_ids.shape[0]
            if self.num_replicas == 1 and self.control_world_size > 1:
                if ratio < 5:
                    batch_size_ratio = (
                        self.control_first_bs * self.control_world_size)
                else:
                    control_base_elements = (
                        self.base_im_w * self.base_im_h
                        * self.control_first_bs)
                    control_batch_size = min(
                        self.max_bs,
                        int(max(
                            1,
                            control_base_elements /
                            (self.base_im_h * ratio * self.base_im_h))))
                    batch_size_ratio = (
                        control_batch_size * self.control_world_size)
            elif ratio < 5:
                batch_size_ratio = self.base_batch_size
            else:
                batch_size_ratio = min(
                    self.max_bs,
                    int(
                        max(1, (self.base_elements /
                                (self.base_im_h * ratio * self.base_im_h)))))
            if num_ratio > batch_size_ratio:
                batch_num_ratio = num_ratio // batch_size_ratio
                print(self.rank, num_ratio, ratio * self.base_im_h,
                      batch_num_ratio, batch_size_ratio)
                ratio_ids_full = ratio_ids[:batch_num_ratio *
                                           batch_size_ratio].reshape(
                                               batch_num_ratio,
                                               batch_size_ratio, 1)
                w = np.full_like(ratio_ids_full, ratio * self.base_im_h)
                h = np.full_like(ratio_ids_full, self.base_im_h)
                ra_wh = np.full_like(ratio_ids_full, ratio)
                ratio_ids_full = np.concatenate([w, h, ratio_ids_full, ra_wh],
                                                axis=-1)
                batch_ratio = ratio_ids_full.tolist()

                if batch_num_ratio * batch_size_ratio < num_ratio:
                    drop = ratio_ids[batch_num_ratio * batch_size_ratio:]
                    if self.is_training:
                        drop_full = ratio_ids[:batch_size_ratio - (
                            num_ratio - batch_num_ratio * batch_size_ratio)]
                        drop = np.append(drop_full, drop)
                    drop = drop.reshape(-1, 1)
                    w = np.full_like(drop, ratio * self.base_im_h)
                    h = np.full_like(drop, self.base_im_h)
                    ra_wh = np.full_like(drop, ratio)

                    drop = np.concatenate([w, h, drop, ra_wh], axis=-1)

                    batch_ratio.append(drop.tolist())
                # Full batches must also be retained when there is no
                # remainder. The upstream indentation dropped an entire
                # aspect-ratio bucket whenever num_ratio was exactly divisible
                # by batch_size_ratio.
                batch_list += batch_ratio
            else:
                print(self.rank, num_ratio, ratio * self.base_im_h,
                      batch_size_ratio)
                ratio_ids = ratio_ids.reshape(-1, 1)
                w = np.full_like(ratio_ids, ratio * self.base_im_h)
                h = np.full_like(ratio_ids, self.base_im_h)
                ra_wh = np.full_like(ratio_ids, ratio)

                ratio_ids = np.concatenate([w, h, ratio_ids, ra_wh], axis=-1)
                batch_list.append(ratio_ids.tolist())
        return batch_list

    def __iter__(self):
        if self.shuffle or self.is_training:
            random.seed(self.epoch)
            if self.dataset_sample_weights is not None:
                self.indices_rank_i_ori, self.distributed_padding = (
                    self._weighted_rank_indices(self.epoch)
                )
                self.indices_rank_i_ratio = self.wh_ratio[self.indices_rank_i_ori]
                self.indices_rank_i_ratio_unique = np.unique(
                    self.indices_rank_i_ratio).tolist()
                self.n_samples_per_replica = len(self.indices_rank_i_ori)
            self.epoch += 1
            self.batch_list = self.create_batch()
            # Rebuild the canonical order before epoch-specific shuffling.
            # Otherwise a resumed process shuffles an ordered list while an
            # uninterrupted process shuffles the previous epoch's permutation,
            # so the same epoch seed does not reproduce the same traversal.
            self.batchs_in_one_epoch_id = [i for i in range(len(self.batch_list))]
            random.shuffle(self.batchs_in_one_epoch_id)
        for batch_tuple_id in self.batchs_in_one_epoch_id:
            yield self.batch_list[batch_tuple_id]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.length
