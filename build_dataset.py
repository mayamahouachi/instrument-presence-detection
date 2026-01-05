#build_dataset.py
"""
Build a segment-level multi-label instrument-presence dataset from MUSDB18  (.stem.mp4).

Output per track:
  - X: (N, n_mels, frames) float32  -> log-mel spectrogram of MIX windows
  - Y: (N, 4) int8                 -> [vocals, drums, bass, other] presence labels per window
  - T: (N,) float32                -> window start times (seconds)

 python build_dataset.py `
>>   --musdb-root "C:\Users\mayam\OneDrive\Documents\3A\IA Musical\projet\musdb18" `
>>   --out-dir "prepared_musdb18" `
>>   --sr 22050 --win-sec 1.0 --hop-sec 0.5 `
>>   --n-mels 64 --ratio-thr 0.05

"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm

import musdb

STEMS = ["vocals", "drums", "bass", "other"]


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.astype(np.float32)
    if x.shape[0] <= 4 and x.shape[1] > 1000: # (channels, samples)
        x = x.T   # (samples,channels)
    return x.mean(axis=1).astype(np.float32) # (samples,)

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x, dtype=np.float64) + 1e-12))


def logmel(y: np.ndarray, sr: int, n_fft: int, hop_length: int, n_mels: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32) # (n_mels, frames) log scale


def build_track_npz(track,out_path: Path,sr: int,win_sec: float,hop_sec: float,n_fft: int,mel_hop: int,n_mels: int,ratio_thr: float,min_mix_rms: float) -> dict:
    """
    Return stats dict about windows saved.
    """
    mix = to_mono(track.audio)
    stems = {name: to_mono(track.targets[name].audio) for name in STEMS}

    # Resample (musdb 44100)
    orig_sr = int(track.rate)
    if orig_sr != sr:
        mix = librosa.resample(mix, orig_sr=orig_sr, target_sr=sr)
        for name in STEMS:
            stems[name] = librosa.resample(stems[name], orig_sr=orig_sr, target_sr=sr)

    win = int(round(win_sec * sr))
    hop = int(round(hop_sec * sr))

    X_list, Y_list, T_list = [], [], []

    total_windows = 0
    kept_windows = 0

    for start in range(0, len(mix) - win + 1, hop):
        total_windows += 1
        end = start + win
        mix_w = mix[start:end]
        mix_r = rms(mix_w)
        y = []
        for name in STEMS:
            stem_w = stems[name][start:end]
            r = rms(stem_w)
            # relative energy criterion
            present = 1 if (r / (mix_r + 1e-12)) >= ratio_thr else 0
            y.append(present)
        feat = logmel(mix_w, sr=sr, n_fft=n_fft, hop_length=mel_hop, n_mels=n_mels) #(n_mels, frames)
        X_list.append(feat)
        Y_list.append(y)
        T_list.append(start / sr)
        kept_windows += 1

    if kept_windows == 0:
        return {"saved": False, "total_windows": total_windows, "kept_windows": 0}

    X = np.stack(X_list, axis=0)  # (N, n_mels, frames)
    Y = np.array(Y_list, dtype=np.int8)  # (N, 4)
    T = np.array(T_list, dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,X=X,Y=Y,T=T,sr=np.int32(sr),win_sec=np.float32(win_sec),
        hop_sec=np.float32(hop_sec),n_fft=np.int32(n_fft),mel_hop=np.int32(mel_hop),n_mels=np.int32(n_mels),
        ratio_thr=np.float32(ratio_thr),min_mix_rms=np.float32(min_mix_rms),track_name=str(track.name))
    return {"saved": True,"total_windows": total_windows,"kept_windows": kept_windows,"X_shape": tuple(X.shape),"Y_shape": tuple(Y.shape)}


def sanity_report(npz_files: list[Path]) -> None:
    if not npz_files:
        print("No files found ")
        return
    counts = np.zeros(4, dtype=np.int64)
    total = 0
    for f in npz_files[:50]:
        d = np.load(f)
        Y = d["Y"]
        counts += Y.sum(axis=0)
        total += Y.shape[0]
    rates = counts / max(total, 1)
    print("Sanity check (approx presence rates on up to 50 files):")
    for i, name in enumerate(STEMS):
        print(f"  {name:6s}: {rates[i]:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--musdb-root", type=str, required=True, help="Path to musdb18 folder containing train/ and test/")
    p.add_argument("--out-dir", type=str, default="prepared_musdb18", help="Output folder")
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--win-sec", type=float, default=1.0)
    p.add_argument("--hop-sec", type=float, default=0.5)
    # Feature params
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--mel-hop", type=int, default=256)
    # Label params
    p.add_argument("--ratio-thr", type=float, default=0.05, help="RMS threshold")
    p.add_argument("--min-mix-rms", type=float, default=1e-4, help="Skip silent windows")
    args = p.parse_args()
    musdb_root = Path(args.musdb_root)
    out_dir = Path(args.out_dir)
    for subset in ["train", "test"]:
        subset_out = out_dir / subset
        subset_out.mkdir(parents=True, exist_ok=True)
        print(f"\nLoading MUSDB subset={subset} from {musdb_root}")
        db = musdb.DB(root=str(musdb_root), subsets=subset)
        saved = 0
        skipped = 0
        tot_kept = 0
        for track in tqdm(db, desc=f"Processing {subset}", unit="track"):
            safe_name = track.name.replace("/", "_").replace("\\", "_")
            out_path = subset_out / f"{safe_name}.npz"
            stats = build_track_npz(track=track,out_path=out_path,sr=args.sr,win_sec=args.win_sec,hop_sec=args.hop_sec,
                n_fft=args.n_fft,mel_hop=args.mel_hop,n_mels=args.n_mels,ratio_thr=args.ratio_thr,min_mix_rms=args.min_mix_rms)
            if stats["saved"]:
                saved += 1
                tot_kept += stats["kept_windows"]
            else:
                skipped += 1
        print(f"\nSubset {subset} done.")
        print(f"  Saved tracks : {saved}")
        print(f"  Skipped      : {skipped}")
        print(f"  Total windows kept (approx): {tot_kept}")
        # Sanity check on generated files
        npz_files = list(subset_out.glob("*.npz"))
        sanity_report(npz_files)
    print(f" Output in: {out_dir.resolve()}")
if __name__ == "__main__":
    main()
