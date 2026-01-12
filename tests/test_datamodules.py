from pathlib import Path
from typing import List, Tuple, cast

import numpy as np
import pytest
import torch

from src.data.musdb18_datamodule import MUSDB18DataModule
from src.utils import sampling


@pytest.mark.parametrize(
    ["file_sizes", "wanted_size_per_file"],
    [
        ([18, 18, 3, 20], 6),
        ([15, 7, 3, 20], 8),
    ],
)
def test_size_splitting(file_sizes: List[int], wanted_size_per_file: int):
    results = sampling.split_at_maximum(file_sizes, wanted_size_per_file)
    assert results.sum() == wanted_size_per_file * len(file_sizes)
    assert np.all(results <= np.array(file_sizes))


@pytest.mark.slow
@pytest.mark.parametrize(
    ["batch_size", "train_val_test_balanced_nb"],
    [(32, (10, 5, 7)), (128, (20, 10, 15))],
)
def test_mus18db_datamodule(
    batch_size: int, train_val_test_balanced_nb: Tuple[int, int, int]
) -> None:
    """Tests `MUSDB18DataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/MUSDB18/prepared"

    train_val_test_split = cast(
        Tuple[int, int, int],
        tuple(100 * 8 * train_val_test_balanced_nb[k] for k in range(3)),
    )
    dm = MUSDB18DataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        train_val_test_split=train_val_test_split,
    )
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == sum(train_val_test_split)

    batch = next(iter(dm.train_dataloader()))
    x, y, _ = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int8
