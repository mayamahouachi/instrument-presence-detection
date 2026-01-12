import enum
from re import sub
from typing import Iterator, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split


def split_with_same_distribution(
    y: NDArray, train_size: int, test_size: Optional[int], seed: int
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
    :return: shape (U,), with U the undersampled dataset size, and the number of classes
    """
    classes, counts = np.unique(targets, return_counts=True)
    min_count = cast(int, counts.min())
    indices = np.concatenate(
        [
            np.random.default_rng(seed).choice(
                np.where(targets == c)[0], size=min_count, replace=False, shuffle=True
            )
            for c in classes
        ]
    )
    return indices, len(classes)


def undersample(y: NDArray[np.int_], subset_size: int, seed: int) -> NDArray[np.int_]:
    """Return a subset with the same target class distribution.

    :param y: of shape (B,)
    :param subset_size: absolute number of samples in the returned subset
    :param seed: the sampling seed
    :return: the sample indices in the subset (shape (subset_size,))
    """
    if len(y) <= subset_size:
        return y
    balanced_subset, num_classes = balance(y, seed)
    undersampled_subset_size = len(balanced_subset)
    if undersampled_subset_size == subset_size:
        return balanced_subset
    if subset_size < undersampled_subset_size:
        if subset_size < num_classes or undersampled_subset_size - subset_size < num_classes:
            # we assume a uniform sampling in a balanced dataset is likely to give
            # a balanced subset in this condition
            return np.random.default_rng(seed).choice(
                balanced_subset, size=subset_size, replace=False, shuffle=True
            )
        # split with stratification only if subset size is enough important
        return balanced_subset[
            split_with_same_distribution(
                y[balanced_subset],
                subset_size,
                None,
                seed,
            )[0]
        ]
    # undersample in a subset with at least one class less
    mask = np.ones(len(y), dtype=bool)
    mask[balanced_subset] = False
    return np.concatenate(
        [
            balanced_subset,
            np.nonzero(mask)[0][
                undersample(y[mask], subset_size - undersampled_subset_size, seed),
            ],
        ]
    )


def split_at_maximum(file_sizes: Sequence[int], wanted_size_per_file: int) -> NDArray[np.int_]:
    """In each file, pick the wanted size if possible. Else, pick greater sizes for heavier files.
    Balance that at a maximum.

    Pre-condition: wanted_size_per_file*len(file_sizes) must be greater than
    the total of file_sizes
    Post-condition: the total size is wanted_size_per_file*len(file_sizes)
    """
    remaining_total_size = wanted_size_per_file * len(file_sizes)
    file_sizes_arr = np.array(file_sizes)
    order = np.argsort(file_sizes_arr)
    file_nb = len(file_sizes)
    sorted_file_sample_sizes = np.empty_like(file_sizes_arr)
    for i, sorted_file_idx in enumerate(order):
        file_size = min(file_sizes_arr[sorted_file_idx].item(), wanted_size_per_file)
        remaining_total_size -= file_size
        wanted_size_per_file = (
            remaining_total_size // (file_nb - i - 1) if i < file_nb - 1 else remaining_total_size
        )
        sorted_file_sample_sizes[i] = file_size
    file_sample_sizes = np.empty_like(file_sizes_arr)
    file_sample_sizes[order] = sorted_file_sample_sizes
    return file_sample_sizes
