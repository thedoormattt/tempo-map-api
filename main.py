import io
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os

app = FastAPI(title="Tempo Map API", version="1.0.0")

# ─── CORS ────────────────────────────────────────────────────────────────────
# In production, restrict to your Vercel domain via the ALLOWED_ORIGINS env var.
# e.g. ALLOWED_ORIGINS=https://tempo-map.vercel.app
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ─── Response models ─────────────────────────────────────────────────────────

class BPMPoint(BaseModel):
    t: float
    bpm: float

class KeyPoint(BaseModel):
    t: float
    key: str
    root: int
    mode: str       # "major" | "minor"
    confidence: float

class KeyResult(BaseModel):
    label: str
    root: int
    mode: str
    confidence: float

class AnalysisResult(BaseModel):
    bpm_curve:   List[BPMPoint]
    key_curve:   List[KeyPoint]
    overall_key: KeyResult
    duration:    float
    sample_rate: int
    channels:    int

# ─── DSP ─────────────────────────────────────────────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def pearson_correlate_key(chroma: np.ndarray):
    """
    Correlate a 12-element chroma vector against all 24 key profiles.
    Returns (key_label, root_index, mode, confidence).
    """
    best_label = "C major"
    best_root  = 0
    best_mode  = "major"
    best_score = -np.inf

    for root in range(12):
        for mode, profile in [("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)]:
            # Rotate profile to this root
            rotated = np.roll(profile, root)
            # Pearson correlation
            score = np.corrcoef(chroma, rotated)[0, 1]
            if score > best_score:
                best_score = score
                best_root  = root
                best_mode  = mode
                best_label = f"{NOTE_NAMES[root]} {mode}"

    return best_label, best_root, best_mode, float(best_score)


def compute_bpm_curve(y: np.ndarray, sr: int, hop_sec: float = 0.5, window_sec: float = 4.0):
    """
    Sliding-window tempo estimation using librosa's beat tracker.
    Much more accurate than the JS autocorrelation approach.
    """
    hop_length = int(sr * hop_sec)
    win_length = int(sr * window_sec)
    results    = []

    # librosa onset envelope — used as input to tempo estimator
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onset_sr  = sr / 512  # onset envelope sample rate

    win_frames = int(window_sec * onset_sr)
    hop_frames = int(hop_sec * onset_sr)

    for i in range(0, len(onset_env) - win_frames, hop_frames):
        window = onset_env[i : i + win_frames]
        # librosa.feature.tempo returns array; take first value
        tempo = librosa.feature.tempo(onset_envelope=window, sr=sr, hop_length=512)[0]
        t     = (i / onset_sr) + window_sec / 2
        results.append({"t": round(t, 3), "bpm": round(float(tempo), 2)})

    return results


def compute_key_curve(y: np.ndarray, sr: int, window_sec: float = 8.0, hop_sec: float = 2.0):
    """
    Sliding-window key estimation using chroma features + Krumhansl-Schmuckler.
    """
    win_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    results     = []

    for start in range(0, len(y) - win_samples, hop_samples):
        segment = y[start : start + win_samples]
        # Chromagram: CQT-based for better pitch accuracy
        chroma  = librosa.feature.chroma_cqt(y=segment, sr=sr)
        # Mean across time → 12-element vector
        chroma_mean = chroma.mean(axis=1)
        label, root, mode, confidence = pearson_correlate_key(chroma_mean)
        t = start / sr + window_sec / 2
        results.append({
            "t":          round(t, 3),
            "key":        label,
            "root":       root,
            "mode":       mode,
            "confidence": round(confidence, 4),
        })

    return results


def compute_overall_key(y: np.ndarray, sr: int):
    """
    Key of the full track using mean chroma across the entire file.
    """
    chroma      = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    label, root, mode, confidence = pearson_correlate_key(chroma_mean)
    return {"label": label, "root": root, "mode": mode, "confidence": round(confidence, 4)}


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyse", response_model=AnalysisResult)
async def analyse(file: UploadFile = File(...)):
    # Validate content type loosely — browsers may send audio/mpeg, audio/wav, etc.
    if file.content_type and not file.content_type.startswith("audio/"):
        # Some browsers send application/octet-stream for certain formats; allow it.
        if file.content_type not in ("application/octet-stream",):
            raise HTTPException(
                status_code=415,
                detail=f"Expected an audio file, got {file.content_type}",
            )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        # librosa.load handles MP3, WAV, FLAC, OGG, M4A etc. via soundfile/audioread
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {e}")

    duration = float(len(y) / sr)
    channels = 1  # librosa always returns mono when mono=True

    try:
        bpm_curve   = compute_bpm_curve(y, sr)
        overall_key = compute_overall_key(y, sr)
        key_curve   = compute_key_curve(y, sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return {
        "bpm_curve":   bpm_curve,
        "key_curve":   key_curve,
        "overall_key": overall_key,
        "duration":    round(duration, 3),
        "sample_rate": int(sr),
        "channels":    channels,
    }
