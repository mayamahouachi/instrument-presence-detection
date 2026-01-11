from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PreparedAudio:
    X: NDArray
    Y: NDArray
    T: NDArray
    sr: int
    win_sec: float
    hop_sec: float
    n_fft: int
    mel_hop: int
    n_mels: int
    stem_abs_thr: float
    ratio_thr_vocals: float
    ratio_thr_drums: float
    ratio_thr_bass: float
    min_mix_rms: float
    track_name: str
