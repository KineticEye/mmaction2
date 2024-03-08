"""COPYRIGHT (C) [2024], CompScience, Inc.

This software is proprietary and confidential. Unauthorized copying,
distribution, modification, or use is strictly prohibited. This software
is provided "as is," without warranty of any kind.
"""
from collections.abc import Sized
from typing import Iterator, Optional

import torch
from torch.utils.data import WeightedRandomSampler
from mmaction.registry import DATA_SAMPLERS
from mmengine.dataset.sampler import DefaultSampler


@DATA_SAMPLERS.register_module()
class WeightedSampler(DefaultSampler):
    def __init__(self,
                 dataset: Sized,
                 replacement: bool = True,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        super().__init__(dataset=dataset, shuffle=shuffle, seed=seed, round_up=round_up)

        # Get per sample weights required by WeightedRandomSampler
        self.weights = self.dataset.per_sample_weights
        self.replacement = replacement

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            if self.weights is None:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(WeightedRandomSampler(
                    weights=self.weights,
                    num_samples=len(self.dataset),
                    replacement=self.replacement,
                    generator=g,
                ))
                with open('indices-weighted-v2-30ep.txt', 'a') as f:
                    for idx in indices:
                        f.write(f'{idx}\n')

        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)
