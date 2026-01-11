from pathlib import Path
from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset, Subset
from typing_extensions import override

from src.data.components.data_type import PreparedAudio
from src.utils import sampling


class MUSDB18Dataset(Dataset[Tuple[Tensor, Tensor]]):
    """Call the subsample or split_with_same_distribution methods to fully load the dataset.

    With those methods, the balancing of the distribution of music pieces and source classes is
    respected at a maximum that depend the wanted absolute number of samples.
    """

    def __init__(self, data_dir: Path, train: bool, transforms, target_transforms, seed=42):
        """
        X: (1, n_mels, frames)
        Y: (3,)  multi-label (vocals, drums, bass)
        """
        self.seed = seed

        data_dir = data_dir / ("train" if train else "test")
        assert data_dir.exists() and data_dir.is_dir(), f"{data_dir} is not directory."
        self.files = list(data_dir.iterdir())

        def get_from_file(file: Path):
            data = PreparedAudio(**np.load(file))
            return data.X, data.T

        self.ready_data = lambda: cast(
            Tuple[Tuple[NDArray, ...], Tuple[NDArray, ...]],
            zip(*(get_from_file(file) for file in self.files)),
        )
        self.data: Optional[NDArray] = None
        self.target_data: Optional[NDArray] = None
        self.transforms = transforms
        self.target_transforms = target_transforms

    def split_with_same_distribution(
        self, train_size: int, test_size: int
    ) -> Tuple[Subset[Tuple[Tensor, Tensor]], Subset[Tuple[Tensor, Tensor]]]:
        """Same target class distribution in train and test subsets.

        The distribution of music pieces may not the same.
        """
        self.subsample(train_size + test_size)
        train_indices, test_indices = sampling.split_with_same_distribution(
            cast(NDArray, self.target_data), train_size, test_size, self.seed
        )
        return Subset(self, train_indices.tolist()), Subset(self, test_indices.tolist())

    def subsample(self, dataset_size: int):
        """Balance according to the number of musical pieces, then balance at a maximum the target
        class representatin for each musical piece."""
        inpts, targets = self.ready_data()
        nb_files = len(inpts)
        sample_nb_per_music = dataset_size // nb_files

        def gen(inpts: Tuple[NDArray, ...], targets: Tuple[NDArray, ...]):
            for inpts_per_file, targets_per_file in zip(inpts, targets):
                indices = sampling.undersample(targets_per_file, sample_nb_per_music, self.seed)
                yield inpts_per_file[indices], targets_per_file[indices]

        inpts, targets = zip(*gen(inpts, targets))
        self.data = np.concatenate(inpts)
        self.target_data = np.concatenate(targets)
        return self

    @override
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        """Select an item (input audio spectrum, target audio source class).

        :return: shape (1, n_mels, frames), shape (3,)
        """
        if self.data is None or self.target_data is None:
            raise Exception(
                "You must call `undersample` or `split_with_same_distribution` method before using the dataset."
            )

        return (
            self.transforms(self.data[index]),
            self.transforms(self.target_data[index]),
        )

    def __len__(self) -> int:
        """Return the number of samples in the given dataset."""
        raise NotImplementedError()
