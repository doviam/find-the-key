"""BPM and key estimation using librosa (beat tracking + Krumhansl–Schmuckler profiles)."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import librosa
import numpy as np

# Krumhansl–Schmuckler key profiles (12 pitch classes, C=0)
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_KEY_NAMES_MAJOR = [
    "C major",
    "C# major",
    "D major",
    "D# major",
    "E major",
    "F major",
    "F# major",
    "G major",
    "G# major",
    "A major",
    "A# major",
    "B major",
]
_KEY_NAMES_MINOR = [
    "C minor",
    "C# minor",
    "D minor",
    "D# minor",
    "E minor",
    "F minor",
    "F# minor",
    "G minor",
    "G# minor",
    "A minor",
    "A# minor",
    "B minor",
]

HOP_LENGTH = 512


@dataclass
class AnalysisResult:
    bpm: int
    bpm_alternates: dict[str, int] = field(default_factory=dict)
    bpm_confidence: float = 0.0
    key: str = "unknown"
    confidence: float = 0.0  # key-estimation confidence


def _preprocess(y: np.ndarray, sr: int, trim_db: float = 30.0) -> np.ndarray:
    """Mono (caller), trim leading/trailing silence, peak normalize."""
    if y.size == 0:
        return y
    y, _ = librosa.effects.trim(y, top_db=trim_db)
    if y.size == 0:
        return y
    y = librosa.util.normalize(y)
    return y


def _bpm_from_beat_track(onset_env: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """Tempo + beat frame indices from onset strength (librosa.beat.beat_track)."""
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=HOP_LENGTH,
        units="frames",
    )
    tempo_flat = np.asarray(tempo, dtype=np.float64).ravel()
    bpm = float(np.median(tempo_flat)) if tempo_flat.size else 120.0
    if bpm <= 0 or np.isnan(bpm):
        try:
            from librosa.feature import rhythm as rhythm_mod

            t2 = rhythm_mod.tempo(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=HOP_LENGTH,
                max_tempo=320,
            )
            bpm = float(np.asarray(t2).ravel()[0])
        except Exception:
            bpm = 120.0
    if bpm <= 0 or np.isnan(bpm):
        bpm = 120.0
    return bpm, np.asarray(beats, dtype=np.int64).ravel()


def _beat_consistency_score(beat_frames: np.ndarray, sr: int) -> float:
    """
    Confidence from inter-beat interval stability (lower coefficient of variation → higher score).
    Returns value in [0, 1].
    """
    if beat_frames.size < 3:
        return 0.25

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    ibi = np.diff(beat_times)
    ibi = ibi[ibi > 1e-6]
    if ibi.size < 2:
        return 0.3

    mean_ibi = float(np.mean(ibi))
    if mean_ibi <= 0:
        return 0.3

    cv = float(np.std(ibi) / mean_ibi)
    # cv ~ 0.02–0.05: very steady; cv > 0.25: loose
    score = 1.0 / (1.0 + 8.0 * cv)
    # Penalize very few beats (short clips)
    n_penalty = min(1.0, beat_frames.size / 32.0)
    score = float(np.clip(score * (0.5 + 0.5 * n_penalty), 0.0, 1.0))
    return round(score, 3)


def _half_double_alternates(bpm_rounded: int) -> dict[str, int]:
    """If detected BPM is in a dubious band, expose the other common interpretation."""
    out: dict[str, int] = {}
    if bpm_rounded < 90:
        out["double_time"] = int(round(bpm_rounded * 2))
    if bpm_rounded > 160:
        out["half_time"] = max(1, int(round(bpm_rounded / 2)))
    return out


def _estimate_key(chroma: np.ndarray) -> tuple[str, float]:
    chroma_mean = np.mean(chroma, axis=1)
    s = float(np.sum(chroma_mean))
    if s > 0:
        chroma_mean = chroma_mean / s
    best_corr = -1.0
    best_name = "C major"
    for i in range(12):
        major_profile = np.roll(_MAJOR, i)
        minor_profile = np.roll(_MINOR, i)
        c_maj = float(np.corrcoef(chroma_mean, major_profile)[0, 1])
        c_min = float(np.corrcoef(chroma_mean, minor_profile)[0, 1])
        if np.isnan(c_maj):
            c_maj = -1.0
        if np.isnan(c_min):
            c_min = -1.0
        if c_maj > best_corr:
            best_corr = c_maj
            best_name = _KEY_NAMES_MAJOR[i]
        if c_min > best_corr:
            best_corr = c_min
            best_name = _KEY_NAMES_MINOR[i]
    return best_name, max(0.0, min(1.0, (best_corr + 1) / 2))


def analyze_audio_bytes(data: bytes, filename: str = "audio") -> AnalysisResult:
    """Load audio from bytes and return BPM + key with pre-processing and beat confidence."""
    y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
    if y.size == 0:
        return AnalysisResult(bpm=0, bpm_confidence=0.0, key="unknown", confidence=0.0)

    y = _preprocess(y, sr)
    if y.size == 0:
        return AnalysisResult(bpm=0, bpm_confidence=0.0, key="unknown", confidence=0.0)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    bpm_float, beat_frames = _bpm_from_beat_track(onset_env, sr)
    bpm_rounded = int(round(bpm_float))
    if bpm_rounded < 1:
        bpm_rounded = 1

    bpm_conf = _beat_consistency_score(beat_frames, sr)
    alternates = _half_double_alternates(bpm_rounded)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    key_name, key_conf = _estimate_key(chroma)

    return AnalysisResult(
        bpm=bpm_rounded,
        bpm_alternates=alternates,
        bpm_confidence=bpm_conf,
        key=key_name,
        confidence=round(key_conf, 3),
    )
