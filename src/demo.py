# demo.py
import time
from typing import Tuple, cast

import hydra
import librosa
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import sounddevice as sd
import soundfile as sf
import torch
from lightning import LightningModule
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from omegaconf import DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras

log = RankedLogger(__name__, rank_zero_only=True)

SR = 22050
WIN_SEC = 0.25
HOP_SEC = 0.25
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


def to_output_device_sr(y: NDArray, sr: int) -> Tuple[NDArray, int]:
    """Resample the audio ndarray of given sr in the device's output sr."""
    out_device = cast(dict, sd.query_devices(kind="output"))
    out_device_sr = int(out_device["default_samplerate"])

    if sr != out_device_sr:
        resampled = librosa.resample(y, orig_sr=sr, target_sr=out_device_sr)
    else:
        resampled = y
    return resampled, out_device_sr


def logmel(y: np.ndarray) -> np.ndarray:
    """Return the logmel of the audio."""
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def load_model(cfg: DictConfig):
    """Load the model checkpoint according to the hydra config."""
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    # WARN: hardcoded
    stems = ["vocals", "drums", "bass"]
    threshold = 0.5
    model.setup("test")
    return model, stems, threshold


def live_demo(y, times: np.ndarray, probs_all: np.ndarray, stems, threshold: float):
    """Stream plots along the music playing."""

    plt.ion()
    n_classes = len(stems)

    fig, ax = cast(Tuple[Figure, Axes], plt.subplots(figsize=(6, 3)))
    x = np.arange(n_classes)
    bars = ax.bar(x, probs_all[0], tick_label=stems)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilité")
    ax.set_title("Présence des instruments")

    for i in range(n_classes):
        ax.hlines(threshold, i - 0.4, i + 0.4, linestyles="dashed")  # ligne de seuil

    fig.tight_layout()

    resampled, out_sr = to_output_device_sr(y, SR)
    sd.play(resampled, out_sr, blocking=False)
    start = time.time()

    for k, t0 in enumerate(times):
        while (time.time() - start) < float(t0):
            time.sleep(0.002)

        probs = probs_all[k]

        for i, b in enumerate(bars):
            b.set_height(probs[i])
            b.set_color("tab:green" if probs[i] >= threshold else "tab:gray")

        ax.set_xlabel(f"Temps : {t0:5.2f} s")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    sd.wait()
    plt.ioff()
    plt.show()


@hydra.main(version_base="1.3", config_path="../configs", config_name="demo.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for the demo.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    assert cfg.ckpt_path
    assert cfg.audio

    model, stems, threshold = load_model(cfg)
    print("Classes :", stems)
    print(f"Seuil global : {threshold:.2f}\n")

    y, sr0 = sf.read(cfg.audio)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr0 != SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)

    win = int(WIN_SEC * SR)
    hop = int(HOP_SEC * SR)
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))

    n_steps = 1 + (len(y) - win) // hop
    times = np.array([k * HOP_SEC for k in range(n_steps)], dtype=np.float32)

    probs_all = np.zeros((n_steps, len(stems)), dtype=np.float32)

    print("Pré-calcul des prédictions...")
    for k in tqdm(range(n_steps), desc="Inference", leave=False):
        s = k * hop
        seg = y[s : s + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))

        X = logmel(seg)  # (n_mels, frames)
        xt = torch.from_numpy(X)[None, None].to(model.device)  # (1,1,n_mels,frames)

        with torch.no_grad():
            logits = model(xt)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        probs_all[k] = probs

    print("Lancement de la démo...")
    live_demo(y, times, probs_all, stems, threshold)


if __name__ == "__main__":
    main()
