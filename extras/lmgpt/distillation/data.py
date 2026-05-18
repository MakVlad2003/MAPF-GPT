"""Arrow streaming dataset matching ``mapf_gpt.fast_data_loader.MapfArrowDataset``.

Same on-disk layout (``*.arrow`` files written by ``generate_dataset.py``), but
supports ``max_files`` so we can iterate on a slice for debug / dry-runs.
"""
from __future__ import annotations

import glob
import os
from collections.abc import Iterator
from typing import Optional, Tuple

import numpy as np
import pyarrow as pa
import torch


class DistillArrowIterable:
    """Streaming Arrow loader: yields (input_tensor, target_tensor) pinned on device."""

    def __init__(
        self,
        folder_path: str,
        device: torch.device,
        batch_size: int,
        max_files: Optional[int] = None,
        seed_shuffle: int = 1337,
    ):
        paths = sorted(glob.glob(os.path.join(folder_path, "*.arrow")))
        if max_files is not None:
            paths = paths[:max_files]
        if not paths:
            raise FileNotFoundError(f"No .arrow files found under {folder_path!r}")
        self.all_data_files = paths
        self.file_paths = paths
        self.device = device
        self.batch_size = batch_size
        self.dtype = torch.int8
        self._rng = np.random.RandomState(seed_shuffle)

        sample_input, _ = self._read_arrow(self.file_paths[0])
        self.input_tensors = torch.empty(sample_input.shape, dtype=self.dtype, device=self.device)
        self.target_tensors = torch.full(sample_input.shape, -1, dtype=self.dtype, device=self.device)

    def _read_arrow(self, file_path: str):
        with pa.memory_map(file_path) as source:
            table = pa.ipc.open_file(source).read_all()
            input_tensors = table["input_tensors"].to_numpy(zero_copy_only=False)
            gt_actions = table["gt_actions"].to_numpy(zero_copy_only=False)
        idx = self._rng.permutation(len(input_tensors))
        input_tensors = np.stack(input_tensors[idx])
        gt_actions = gt_actions[idx]
        return input_tensors, gt_actions

    def load_and_transfer_data_file(self, filename: str) -> None:
        input_tensors, gt_actions = self._read_arrow(filename)
        self.input_tensors.copy_(torch.tensor(input_tensors, dtype=self.dtype), non_blocking=True)
        self.target_tensors[:, -1].copy_(torch.tensor(gt_actions, dtype=self.dtype), non_blocking=True)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            for file_path in self.file_paths:
                self.load_and_transfer_data_file(file_path)
                for i in range(0, len(self.input_tensors), self.batch_size):
                    yield (
                        self.input_tensors[i: i + self.batch_size],
                        self.target_tensors[i: i + self.batch_size],
                    )

    def get_full_dataset_size(self) -> int:
        return len(self.input_tensors) * len(self.all_data_files)


__all__ = ["DistillArrowIterable"]
