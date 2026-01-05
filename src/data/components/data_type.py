from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class PreparedAudio(TypedDict):
    X: NDArray
    Y: NDArray
    T: NDArray
    sr: NDArray[np.int32]
    win_sec: NDArray[np.float32]
    hop_sec: NDArray[np.float32]
    n_fft: NDArray[np.int32]
    mel_hop: NDArray[np.int32]
    n_mels: NDArray[np.int32]
    ratio_thr: NDArray[np.float32]
    min_mix_rms: NDArray[np.float32]
    track_name: str
