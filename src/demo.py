import argparse
import time
import os, sys
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from hydra import initialize, compose
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT)

# Same params as dataset 
SR = 22050
WIN_SEC = 0.25
HOP_SEC = 0.25
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

STEMS = ["vocals", "drums", "bass"]
THRESHOLD = 0.5
COLORS = ["tab:orange", "tab:blue", "tab:green"]


def logmel(y: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP_LENGTH,n_mels=N_MELS,power=2.0)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="eval", overrides=[f"ckpt_path={ckpt_path}"])

    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def live_demo(y, times, probs_all):
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(STEMS))
    bars = ax.bar(x, probs_all[0], tick_label=STEMS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")

    for i in range(len(STEMS)):
        ax.hlines(THRESHOLD, i - 0.4, i + 0.4, linestyles="dashed")

    sd.play(y, SR, blocking=False)
    start = time.time()

    for k, t0 in enumerate(times):
        while (time.time() - start) < float(t0):
            time.sleep(0.002)

        probs = probs_all[k]
        best = np.argmax(probs)

        print(f"[{t0:5.2f}s] vocals={probs[0]:.3f} drums={probs[1]:.3f} bass={probs[2]:.3f} --> Dominant: {STEMS[best].upper()}")

        # Update bars
        for i, b in enumerate(bars):
            b.set_height(probs[i])
            if probs[i] >= THRESHOLD:
                b.set_color(COLORS[i])
            else:
                b.set_color("lightgray")


        ax.set_title(f"{t0:5.2f}s  {STEMS[best].upper()} ")
        ax.set_xlabel("Instruments Detected")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    sd.wait()
    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Live Music Instrument Detection Demo")
    parser.add_argument("--audio", required=True, help="Chemin vers le fichier audio (.wav)")
    parser.add_argument("--ckpt", required=True, help="Chemin vers le modèle .ckpt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n Device utilisé :", device)

    # Load model
    model = load_model(Path(args.ckpt), device)
    print(" Modèle chargé ! Classes :", STEMS, "\n")

    # Load audio
    y, sr0 = sf.read(args.audio)
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
    probs_all = np.zeros((n_steps, len(STEMS)), dtype=np.float32)

    print(" Pré-calcul des prédictions...\n")
    for k in tqdm(range(n_steps), desc="Inference"):
        s = k * hop
        seg = y[s : s + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))

        X = logmel(seg)
        xt = torch.from_numpy(X)[None, None].to(device)

        with torch.no_grad():
            probs = torch.sigmoid(model(xt)).cpu().numpy()[0]
        probs_all[k] = probs

    print("Lancement de la démo en temps réel !\n")
    live_demo(y, times, probs_all)


if __name__ == "__main__":
    main()
