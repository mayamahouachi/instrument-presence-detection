# build_dataset.py
"""Build a segment-level multi-label instrument-presence dataset from MUSDB18  (.stem.mp4).

Output per track:
  - X: (N, n_mels, frames)  -> log-mel spectrogram of MIX windows
  - Y: (N, 3)               -> [vocals, drums, bass] presence labels per window
  - T: (N,)                 -> window start times (seconds)

python build_dataset.py \
  --musdb-root "/path/to/musdb18" \
  --out-dir "prepared_musdb18" \
>>   --sr 22050 --win-sec 1.0 --hop-sec 0.5 `
>>   --n-mels 64 --n-fft 1024 --mel-hop 256 `
>>   --ratio-thr-vocals 0.07 `
>>   --ratio-thr-drums 0.12 `
>>   --ratio-thr-bass 0.12 `
>>   --min-mix-rms 1e-4 `
>>   --stem-abs-thr 1e-3
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, cast

import librosa
import musdb
import numpy as np
from tqdm import tqdm

STEMS = ["vocals", "drums", "bass"]


def to_mono(x: np.ndarray) -> np.ndarray:
    """Convert the multi-channel signal into a signal with one channel."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x.astype(np.float32)
    if x.shape[0] <= 4 and x.shape[1] > 1000:  # (channels, samples)
        x = x.T  # (samples,channels)
    return x.mean(axis=1).astype(np.float32)  # (samples,)


def rms(x: np.ndarray) -> float:
    """RMS Conversion."""
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x, dtype=np.float64) + 1e-12))


def logmel(y: np.ndarray, sr: int, n_fft: int, hop_length: int, n_mels: int) -> np.ndarray:
    """Return the mel spectogram.

    See the course or the librosa documentation.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)  # (n_mels, frames) log scale


def build_track_npz(
    track,
    out_path: Path,
    sr: int,
    win_sec: float,
    hop_sec: float,
    n_fft: int,
    mel_hop: int,
    n_mels: int,
    ratio_thr_map: dict[str, float],
    min_mix_rms: float,
    stem_abs_thr: float,
    compress: bool,
) -> dict:
    """Return stats dict about windows saved."""
    mix = to_mono(track.audio)
    stems = {name: to_mono(track.targets[name].audio) for name in STEMS}

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
    skipped_silent = 0
    for start in range(0, len(mix) - win + 1, hop):
        total_windows += 1
        end = start + win
        mix_w = mix[start:end]
        mix_r = rms(mix_w)
        if mix_r < min_mix_rms:
            skipped_silent += 1
            continue
        y = []
        denom = mix_r + 1e-12
        for name in STEMS:
            stem_w = stems[name][start:end]
            r = rms(stem_w)
            present = 1 if (r >= stem_abs_thr and (r / denom) >= ratio_thr_map[name]) else 0
            y.append(present)
        feat = logmel(mix_w, sr=sr, n_fft=n_fft, hop_length=mel_hop, n_mels=n_mels)
        X_list.append(feat)
        Y_list.append(y)
        T_list.append(start / sr)
        kept_windows += 1

    if kept_windows == 0:
        return {
            "saved": False,
            "total_windows": total_windows,
            "kept_windows": 0,
            "skipped_silent": skipped_silent,
        }

    X = np.stack(X_list, axis=0)  # (N, n_mels, frames)
    Y = np.array(Y_list, dtype=np.int8)  # (N, 3)
    T = np.array(T_list, dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        X=X,
        Y=Y,
        T=T,
        sr=sr,
        win_sec=win_sec,
        hop_sec=hop_sec,
        n_fft=n_fft,
        mel_hop=mel_hop,
        n_mels=n_mels,
        min_mix_rms=min_mix_rms,
        stem_abs_thr=stem_abs_thr,
        ratio_thr_vocals=ratio_thr_map["vocals"],
        ratio_thr_drums=ratio_thr_map["drums"],
        ratio_thr_bass=ratio_thr_map["bass"],
        track_name=str(track.name),
    )
    if compress:
        np.savez_compressed(out_path, **payload)
    else:
        np.savez(out_path, **payload)
    return {
        "saved": True,
        "total_windows": total_windows,
        "kept_windows": kept_windows,
        "skipped_silent": skipped_silent,
        "X_shape": tuple(X.shape),
        "Y_shape": tuple(Y.shape),
    }


def sanity_report(npz_files: list[Path], max_files: int = 50) -> None:
    """Log the presence rate of each stem over the batch of audio files."""
    if not npz_files:
        print("No files found ")
        return
    counts = np.zeros(3)
    total = 0
    for f in npz_files[:max_files]:
        d = np.load(f)
        Y = d["Y"]
        counts += Y.sum(axis=0)
        total += Y.shape[0]
    rates = counts / max(total, 1)
    print(f"Sanity check (approx presence rates on up to {min(len(npz_files), max_files)} files):")
    for i, name in enumerate(STEMS):
        print(f"  {name:6s}: {rates[i]:.3f}")


def main():
    """Dataset builder CLI entrypoint."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--musdb-root",
        type=str,
        required=True,
        help="Path to musdb18 folder containing train/ and test/",
    )
    p.add_argument("--out-dir", type=str, default="prepared_musdb18", help="Output folder")
    # audio / segmentation
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--win-sec", type=float, default=1.0)
    p.add_argument("--hop-sec", type=float, default=0.5)
    # feature params
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--mel-hop", type=int, default=256)
    # label params
    p.add_argument("--ratio-thr", type=float, default=0.05, help="Default relative RMS threshold")
    p.add_argument("--ratio-thr-vocals", type=float, default=None)
    p.add_argument("--ratio-thr-drums", type=float, default=None)
    p.add_argument("--ratio-thr-bass", type=float, default=None)
    p.add_argument(
        "--min-mix-rms", type=float, default=1e-4, help="Skip silent windows based on mix RMS"
    )
    p.add_argument("--stem-abs-thr", type=float, default=1e-4, help="Absolute RMS gate for stems")
    p.add_argument("--no-compress", action="store_true", help="Save uncompressed .npz")
    args = p.parse_args()
    ratio_thr_map = {
        "vocals": args.ratio_thr if args.ratio_thr_vocals is None else args.ratio_thr_vocals,
        "drums": args.ratio_thr if args.ratio_thr_drums is None else args.ratio_thr_drums,
        "bass": args.ratio_thr if args.ratio_thr_bass is None else args.ratio_thr_bass,
    }

    musdb_root = Path(args.musdb_root)
    out_dir = Path(args.out_dir)
    compress = not args.no_compress
    print("Config:")
    print(f"  sr={args.sr} win={args.win_sec}s hop={args.hop_sec}s")
    print(f"  logmel: n_mels={args.n_mels} n_fft={args.n_fft} mel_hop={args.mel_hop}")
    print(f"  min_mix_rms={args.min_mix_rms} stem_abs_thr={args.stem_abs_thr}")
    print("  ratio_thr_map:", ratio_thr_map)
    print(f"  save={'compressed' if compress else 'uncompressed'}")
    for subset in ["train", "test"]:
        subset_out = out_dir / subset
        subset_out.mkdir(parents=True, exist_ok=True)
        print(f"\nLoading MUSDB subset={subset} from {musdb_root}")
        db = musdb.DB(root=str(musdb_root), subsets=subset)
        saved = 0
        skipped = 0
        tot_kept = 0
        tot_silent = 0
        for track in tqdm(cast(Iterable, db), desc=f"Processing {subset}", unit="track"):
            safe_name = track.name.replace("/", "_").replace("\\", "_")
            out_path = subset_out / f"{safe_name}.npz"
            stats = build_track_npz(
                track=track,
                out_path=out_path,
                sr=args.sr,
                win_sec=args.win_sec,
                hop_sec=args.hop_sec,
                n_fft=args.n_fft,
                mel_hop=args.mel_hop,
                n_mels=args.n_mels,
                ratio_thr_map=ratio_thr_map,
                min_mix_rms=args.min_mix_rms,
                stem_abs_thr=args.stem_abs_thr,
                compress=compress,
            )
            if stats["saved"]:
                saved += 1
                tot_kept += stats["kept_windows"]
                tot_silent += stats["skipped_silent"]
            else:
                skipped += 1
        print(f"\nSubset {subset} done.")
        print(f"  Saved tracks : {saved}")
        print(f"  Skipped      : {skipped}")
        print(f"  Total windows kept (approx): {tot_kept}")
        print(f"  Total windows silent: {tot_silent}")
        npz_files = list(subset_out.glob("*.npz"))
        sanity_report(npz_files)
    print(f" Output in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
