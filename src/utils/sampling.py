from re import sub
from typing import Tuple, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split


def split_with_same_distribution(
    y: NDArray, train_size: int, test_size: int, seed: int
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Return train and valid sample indices.

    :param y: of shape (B,)
    :param test_size: absolute number of test samples
    :param train_size: absolute number of train samples
    :param seed: the sampling seed
    :return: train and test sample indices (of shapes (train_size,) and (test_size,))
    """
    return cast(
        Tuple[NDArray[np.int_], NDArray[np.int_]],
        tuple(
            train_test_split(
                np.arange(len(y)),
                test_size=test_size,
                train_size=train_size,
                stratify=y,
                random_state=seed,
            )
        ),
    )


def balance(targets: NDArray[np.int_], seed: int):
    """Return sample indices for a subset with a uniform distribution of the target classes.

    The undersampling technique is used. Be careful than the number of samples per class may be
    much smaller.

    :param targets: shape (B,)
    :param seed: the sampling seed
    :return: shape (U,), with U the undersampled dataset size
    """
    classes, counts = np.unique(targets)
    min_count = cast(int, counts.min())
    indices = np.concatenate(
        [
            np.random.default_rng(seed).choice(
                np.where(targets == c)[0], size=min_count, replace=False, shuffle=True
            )
            for c in classes
        ]
    )
    return indices


def undersample(y: NDArray[np.int_], subset_size: int, seed: int) -> NDArray[np.int_]:
    """Return a subset with the same target class distribution.

    :param y: of shape (B,)
    :param subset_size: absolute number of samples in the returned subset
    :param seed: the sampling seed
    :return: the sample indices in the subset (shape (subset_size,))
    """
    if len(y) <= subset_size:
        return y
    balanced_subset = balance(y, seed)
    if len(balanced_subset) == subset_size:
        return balanced_subset
    if len(balanced_subset) > subset_size:
        return split_with_same_distribution(balanced_subset, subset_size, 0, seed)[0]
    # undersample in a subset with at least one class less
    mask = np.ones(len(y), dtype=bool)
    mask[balanced_subset] = False
    return np.concatenate(
        [
            balanced_subset,
            undersample(y[mask], subset_size - len(balanced_subset), seed),
        ]
    )
