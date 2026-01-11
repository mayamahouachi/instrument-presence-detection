from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

STEMS = ["vocals", "drums", "bass"]


def list_npz(data_dir: str | Path) -> list[Path]:
    """Return the list of pre-processed files."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz found in: {data_dir}")
    return files


def compute_train_stats(npz_files, max_segments_per_track: int | None = 2000, seed: int = 0):
    """Return the mean and standard deviation of the spectogram values in the train dataset."""

    rng = np.random.default_rng(seed)
    sums: NDArray[np.float_] | None = None
    sq_sums: NDArray[np.float_] | None = None
    count = 0

    for f in npz_files:
        d = np.load(f)
        X = cast(NDArray[np.float_], d["X"])  # (N, n_mels, frames)

        if max_segments_per_track is not None and X.shape[0] > max_segments_per_track:
            idx = rng.choice(X.shape[0], size=max_segments_per_track, replace=False)
            X = X[idx]

        if sums is None:
            sums = X.sum(axis=0)
            sq_sums = (X * X).sum(axis=0)
        else:
            sums += X.sum(axis=0)
            sq_sums += (X * X).sum(axis=0)

        count += X.shape[0]

    mean = cast(NDArray[np.float_], sums) / max(count, 1)
    var = (cast(NDArray[np.float_], sq_sums) / max(count, 1)) - (mean * mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean, std


def save_stats(path: str | Path, mean: float, std: NDArray[np.float_]):
    """Save the normalization data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, std=std)


def load_stats(path: str | Path):
    """Reload the normalization data."""
    d = np.load(Path(path))
    return {"mean": d["mean"], "std": d["std"]}


class SegmentDataset(Dataset):
    """
    X: (1, n_mels, frames)
    Y: (3,)  multi-label (vocals, drums, bass)
    """

    def __init__(
        self,
        npz_files,
        normalize: bool = True,
        stats: dict | None = None,
        max_per_track: int | None = None,
        seed: int = 0,
        cache_files: bool = True,
    ):
        self.files = list(npz_files)
        if not self.files:
            raise ValueError("Empty npz_files.")

        self.normalize = normalize
        self.stats = stats
        self.cache_files = cache_files
        self._cache = {}
        rng = np.random.default_rng(seed)
        self.index = []
        for fi, f in enumerate(self.files):
            d = np.load(f)
            n = int(d["X"].shape[0])
            idxs = np.arange(n)
            if max_per_track is not None and n > max_per_track:
                idxs = rng.choice(idxs, size=max_per_track, replace=False)
            self.index.extend((fi, int(i)) for i in idxs)
        if not self.index:
            raise ValueError("No segments found")

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.index)

    def _load(self, fi: int):
        """Load the the i-th audio file from cache or from disk."""
        f = self.files[fi]
        if self.cache_files:
            if f not in self._cache:
                self._cache[f] = np.load(f)
            return self._cache[f]
        return np.load(f)

    def __getitem__(self, idx: int):
        """Return a sample."""

        fi, si = self.index[idx]
        d = self._load(fi)

        X = d["X"][si]  # (n_mels, frames)
        Y = d["Y"][si]  # (3,)

        if self.normalize and self.stats is not None:
            X = (X - self.stats["mean"]) / (self.stats["std"] + 1e-8)

        X = torch.from_numpy(X).unsqueeze(0)  # (1, n_mels, frames)
        Y = torch.from_numpy(Y)  # (3,)
        return X, Y
