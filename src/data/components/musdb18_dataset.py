from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch import Tensor, tensor
from torch.utils.data import Dataset, Subset
from typing_extensions import override

from src.data.components.data_type import PreparedAudio
from src.utils import sampling
from src.utils.to_class import to_one_integer_class


class MUSDB18Dataset(Dataset[Tuple[Tensor, Tensor, NDArray]]):
    """Call the subsample or split_with_same_distribution methods to fully load the dataset.

    With those methods, the balancing of the distribution of music pieces and source classes is
    respected at a maximum that depend the wanted absolute number of samples.

    An item is the following tuple:
    * mel spectrum windows: shape (1, n_mels, frame) (first dim for 1 kernel)
    * occurring music source classes for the window: shape (3,)
    * source position (file_nb, position_in_file): shape (2,)
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
            return data.X, data.Y

        self.ready_data = lambda: cast(
            Tuple[Tuple[NDArray, ...], Tuple[NDArray, ...]],
            zip(*(get_from_file(file) for file in self.files)),
        )
        self.data: Optional[NDArray] = None
        self.target_data: Optional[NDArray] = None
        self.source_position: Optional[NDArray] = None
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.file_number = len(self.files)

    def _encode_target(self, target: NDArray[np.bool_]) -> NDArray[np.int_]:
        return to_one_integer_class(tensor(target, device="cpu")).numpy()

    def split_with_same_distribution(
        self, train_size: int, test_size: int
    ) -> Tuple[Subset[Tuple[Tensor, Tensor, NDArray]], Subset[Tuple[Tensor, Tensor, NDArray]]]:
        """Same target class distribution in train and test subsets.

        The distribution of music pieces may not the same.
        """
        self.subsample(train_size + test_size)
        train_indices, test_indices = sampling.split_with_same_distribution(
            self._encode_target(cast(NDArray, self.target_data)),
            train_size,
            test_size,
            self.seed,
        )
        return Subset(self, train_indices.tolist()), Subset(self, test_indices.tolist())

    def subsample(self, dataset_size: int):
        """Balance according to the number of musical pieces, then balance at a maximum the target
        class representatin for each musical piece."""
        inpts, targets = self.ready_data()
        nb_files = len(inpts)

        def sample_nb_per_music():
            quotient = dataset_size // nb_files
            return sampling.split_at_maximum(
                [len(target) for target in targets], quotient
            ).tolist()

        def gen(
            inpts: Tuple[NDArray, ...],
            targets: Tuple[NDArray, ...],
            ds_sizes: Iterable[int],
        ):
            for file_nb, (
                inpts_per_file,
                targets_per_file,
                ds_size_per_file,
            ) in enumerate(zip(inpts, targets, ds_sizes)):
                indices = sampling.undersample(
                    self._encode_target(targets_per_file),
                    ds_size_per_file,
                    self.seed,
                )
                source_positions = np.empty((len(indices), 2), dtype=indices.dtype)
                source_positions[:, 0] = file_nb
                source_positions[:, 1] = indices
                yield inpts_per_file[indices], targets_per_file[indices], source_positions

        inpts, targets, positions = zip(*gen(inpts, targets, sample_nb_per_music()))
        self.data = np.concatenate(inpts)[:, np.newaxis, ...]
        self.target_data = np.concatenate(targets)
        self.source_position = np.concatenate(positions)
        return self

    @override
    def __getitem__(self, index) -> Tuple[Tensor, Tensor, NDArray]:
        """Select an item (input audio spectrum, target audio source class).

        :return: shape (1, n_mels, frames), shape (3,)
        """
        if self.data is None or self.target_data is None or self.source_position is None:
            raise Exception(
                "You must call `undersample` or `split_with_same_distribution` method before using the dataset."
            )

        return (
            self.transforms(self.data[index]),
            self.transforms(self.target_data[index]),
            self.source_position[index],
        )

    def __len__(self) -> int:
        """Return the number of samples in the given dataset."""
        if self.data is None or self.target_data is None:
            raise Exception(
                "You must call `undersample` or `split_with_same_distribution` method before using the dataset."
            )
        return len(self.data)
