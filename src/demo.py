# demo.py
import argparse
import time
from pathlib import Path
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from tqdm import tqdm
from model import InstrumentCNN
import matplotlib.pyplot as plt

SR = 22050
WIN_SEC = 0.25
HOP_SEC = 0.25
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

def logmel(y: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP_LENGTH,n_mels=N_MELS,power=2.0)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def load_model(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    stems = ckpt.get("stems", ["vocals", "drums", "bass"])
    threshold = float(ckpt.get("threshold", 0.5))
    model = InstrumentCNN(n_classes=len(stems))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, stems, threshold


def live_demo(y, times: np.ndarray,probs_all: np.ndarray, stems, threshold: float):

    plt.ion()
    n_classes = len(stems)

    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(n_classes)
    bars = ax.bar(x, probs_all[0], tick_label=stems)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilité")
    ax.set_title("Présence des instruments")

    for i in range(n_classes):
        ax.hlines(threshold, i - 0.4, i + 0.4, linestyles="dashed")  # ligne de seuil

    fig.tight_layout()

    sd.play(y, SR, blocking=False)
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


def main():
    parser = argparse.ArgumentParser( description="Démo : détection d'instruments en temps réel sur un extrait audio.")
    parser.add_argument("--audio", required=True, help="Chemin vers le fichier audio")
    parser.add_argument("--model",required=True, help="Chemin vers le modèle .ckpt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)

    model, stems, threshold = load_model(Path(args.model), device)
    print("Classes :", stems)
    print(f"Seuil global : {threshold:.2f}\n")

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

    probs_all = np.zeros((n_steps, len(stems)), dtype=np.float32)

    print("Pré-calcul des prédictions...")
    for k in tqdm(range(n_steps), desc="Inference", leave=False):
        s = k * hop
        seg = y[s:s + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))

        X = logmel(seg) # (n_mels, frames)
        xt = torch.from_numpy(X)[None, None].to(device)  # (1,1,n_mels,frames)

        with torch.no_grad():
            logits = model(xt)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        probs_all[k] = probs

    print("Lancement de la démo...")
    live_demo(y, times, probs_all, stems, threshold)


if __name__ == "__main__":
    main()
