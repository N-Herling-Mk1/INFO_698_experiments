"""Song drop-in -> per-genre scores + uncertainty. Reuses the proven bundle:
load_bundle -> window the audio -> per-window (mel + 58 tabular) -> scale with the
bundle's scaler -> forward -> epistemic σ -> aggregate to song level.

Epistemic σ is MC-dropout here (works with the deterministic backbone TODAY). The
LLLA fast path (Day-2 `bayes` extra) replaces this with the closed-form last-layer
posterior the bundle's ggn_eig.npz precomputes; the /api contract below is unchanged
when that lands — only the σ source swaps.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from .bundle import load_bundle
from . import features as F


def _scale(x_tab: np.ndarray, scaler: dict) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    return ((x_tab - mean) / std).astype(np.float32)


def _enable_mc_dropout(model):
    """Eval mode (BN uses running stats) but Dropout left ON for MC sampling."""
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


@torch.no_grad()
def _mc_forward(model, img, tab, T: int = 30):
    """T stochastic passes -> (mean_probs [C], epistemic_sigma [C]) per window."""
    probs = []
    for _ in range(T):
        probs.append(torch.softmax(model(img, tab), dim=1).cpu().numpy())
    P = np.stack(probs, axis=0)            # [T, N, C]
    return P.mean(0), P.std(0)             # [N, C], [N, C]


def predict_song(audio_path, model_dir="projects/genre/models/beardown",
                 mel_cfg: dict | None = None, image_size: int = 224,
                 mc_samples: int = 30, device: str = "cpu") -> dict:
    """Drop a wav -> song-level per-genre scores + uncertainty + per-window detail."""
    mel_cfg = mel_cfg or {}
    b = load_bundle(model_dir, device=device)
    genres = b.genres
    cols = b.scaler.get("cols")

    y, sr = F.load_audio(audio_path)
    windows = F.window_segments(y, sr, seconds=30.0)

    seg_mean, seg_sig, seg_rows = [], [], []
    for i, w in enumerate(windows):
        mel = F.extract_mel(w, sr, image_size=image_size,
                            n_mels=mel_cfg.get("n_mels", F.N_MELS),
                            n_fft=mel_cfg.get("n_fft", F.N_FFT),
                            hop_length=mel_cfg.get("hop_length", F.HOP))
        tab = F.extract_tabular(w, sr, n_fft=mel_cfg.get("n_fft", F.N_FFT),
                                hop_length=mel_cfg.get("hop_length", F.HOP), cols=cols)
        tab = _scale(tab[None, :], b.scaler)
        img_t = torch.from_numpy(mel[None, None, :, :]).to(device)          # [1,1,H,W]
        tab_t = torch.from_numpy(tab).to(device)                            # [1,F]
        _enable_mc_dropout(b.model)
        mean_p, sig = _mc_forward(b.model, img_t, tab_t, T=mc_samples)
        mean_p, sig = mean_p[0], sig[0]
        seg_mean.append(mean_p); seg_sig.append(sig)
        top = int(mean_p.argmax())
        seg_rows.append({"window": i, "top_genre": genres[top],
                         "top_prob": float(mean_p[top]),
                         "probs": {g: float(p) for g, p in zip(genres, mean_p)}})

    seg_mean = np.stack(seg_mean); seg_sig = np.stack(seg_sig)
    song_prob = seg_mean.mean(0)                                            # [C]
    # aggregate uncertainty: within-window epistemic (mean σ) + across-window spread
    song_sig = np.sqrt((seg_sig**2).mean(0) + seg_mean.var(0))
    order = np.argsort(song_prob)[::-1]

    return {
        "model_dir": str(model_dir),
        "n_windows": len(windows),
        "sigma_source": "mc_dropout",   # -> "llla" once the Day-2 bayes path lands
        "per_genre": [{"genre": genres[i], "prob": float(song_prob[i]),
                       "sigma": float(song_sig[i])} for i in order],
        "top": {"genre": genres[int(order[0])], "prob": float(song_prob[order[0]]),
                "sigma": float(song_sig[order[0]])},
        "segments": seg_rows,
    }


def has_model(model_dir="projects/genre/models/beardown") -> bool:
    d = Path(model_dir)
    return (d / "weights.pt").exists() and (d / "arch.json").exists()


def predict_upload(file_path, exp_root, device: str = "cpu") -> dict:
    """Uniform entrypoint the generic /api/predict route calls (one per experiment).
    Reads the run_name + mel/image config, resolves the bundle, runs predict_song."""
    import yaml
    exp_root = Path(exp_root)
    cfg = yaml.safe_load(open(exp_root / "configs" / "beardown.yaml", encoding="utf-8"))
    model_dir = exp_root / "models" / cfg.get("run_name", "beardown")
    if not has_model(str(model_dir)):
        return {"error": "no trained model yet — run train.py first", "model_dir": str(model_dir)}
    return predict_song(str(file_path), model_dir=str(model_dir),
                        mel_cfg=cfg.get("mel", {}),
                        image_size=cfg.get("features", {}).get("image_size", 224),
                        device=device)
