from pathlib import Path
from typing import Tuple, cast

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import override

from src.data.components.data_type import PreparedAudio


class MUSDB18Dataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, data_dir: Path, train: bool, transforms):
        data_dir = data_dir / ("train" if train else "test")
        assert data_dir.exists() and data_dir.is_dir(), f"{data_dir} is not directory."
        files = data_dir.iterdir()

        def get_from_file(file: Path):
            data = PreparedAudio(**np.load(file))
            return data.X, data.T

        inpts, targets = cast(
            Tuple[Tuple[NDArray, ...], Tuple[NDArray, ...]],
            zip(*(get_from_file(file) for file in files)),
        )
        self.data, self.target_data = (np.concatenate(inpts), np.concatenate(targets))
        self.transforms = transforms

    def balance(self, target_values: NDArray[np.bool_], batch_size: int) -> NDArray[np.int64]:
        """Return a set sampling indices which balance the target value distribution. It also try
        on a second hand to balance at a maximum the presence of the tracks.

        Let I be the track (sound source kind to be guessed) number in an audio. Let T be the
        number of musical pieces.

        :param target_values: shape (T, B, I). The target values, equaling 1 at index (t, b, i) if
            the i-th sound source kind is not in silence in the b-th window of the t-th musical
            piece.
        :return: shape (batch_size, 2). The set of indices in the space of the given dataset ((T,
            B)).
        """
        track_nb = target_values.shape[2]
        target_class_idx = 2 ** np.arange(track_nb)
        raise NotImplementedError()

    @override
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        """Select an item (input audio spectrum, target audio source class)."""
        return (self.transforms(self.data[index]), self.transforms(self.target_data[index]))

    def __len__(self) -> int:
        """Return the number of samples in the given dataset."""
        raise NotImplementedError()
