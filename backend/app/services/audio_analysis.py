"""BPM and key estimation using librosa (beat tracking + Krumhansl–Schmuckler)."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import librosa
import numpy as np

# Krumhansl–Schmuckler key profiles (12 pitch classes, C = index 0).
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)

NOTE_NAMES: list[str] = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]

HOP_LENGTH = 512

# Key-focused window: skip intros / drops with weak harmony (electronic, rap, etc.).
KEY_WINDOW_OFFSET_SEC = 30.0
KEY_WINDOW_DURATION_SEC = 30.0
# If the file is shorter than offset+duration, fall back to the start of the file.
MIN_SEGMENT_SEC = 5.0
# Pre-emphasis for chroma path (clarity in mid–high harmonics).
PREEMPHASIS_COEF = 0.97


def _major_key_label(tonic_index: int) -> str:
    return f"{NOTE_NAMES[tonic_index % 12]} major"


def _minor_key_label(tonic_index: int) -> str:
    return f"{NOTE_NAMES[tonic_index % 12]} minor"


@dataclass
class AnalysisResult:
    bpm: int
    bpm_alternates: dict[str, int] = field(default_factory=dict)
    bpm_confidence: float = 0.0
    key: str = "unknown"
    confidence: float = 0.0  # key-estimation confidence (0–1)


def _load_mono_window(
    data: bytes,
    offset_sec: float,
    duration_sec: float | None,
) -> tuple[np.ndarray, int]:
    """Decode a mono segment; duration None = until end of file."""
    return librosa.load(
        io.BytesIO(data),
        sr=None,
        mono=True,
        offset=offset_sec,
        duration=duration_sec,
    )


def _segment_duration_sec(y: np.ndarray, sr: int) -> float:
    if sr <= 0 or y.size == 0:
        return 0.0
    return float(y.size) / float(sr)


def _load_body_or_fallback(data: bytes) -> tuple[np.ndarray, int]:
    """
    Prefer [30s, 60s] of the track for analysis; if that region is missing or too short,
    use [0s, 30s] so short clips and DJ tools still return a result.
    """
    y_body, sr = _load_mono_window(
        data,
        KEY_WINDOW_OFFSET_SEC,
        KEY_WINDOW_DURATION_SEC,
    )
    need_sec = max(MIN_SEGMENT_SEC, 1.0)
    if _segment_duration_sec(y_body, sr) >= need_sec:
        return y_body, sr

    y_head, sr2 = _load_mono_window(data, 0.0, KEY_WINDOW_DURATION_SEC)
    if y_head.size == 0:
        return y_body, sr
    if y_body.size == 0 or _segment_duration_sec(y_head, sr2) > _segment_duration_sec(y_body, sr):
        return y_head, sr2
    return y_body, sr


def _preprocess(y: np.ndarray, sr: int, trim_db: float = 30.0) -> np.ndarray:
    """Trim leading/trailing silence, peak normalize."""
    if y.size == 0:
        return y
    y, _ = librosa.effects.trim(y, top_db=trim_db)
    if y.size == 0:
        return y
    y = librosa.util.normalize(y)
    return y


def _harmonic_preemphasis(y: np.ndarray, sr: int) -> np.ndarray:
    """
    HPSS → harmonic only → pre-emphasis → re-normalize for stable chroma_cqt.
    """
    if y.size == 0:
        return y
    y_harmonic, _ = librosa.effects.hpss(y)
    if y_harmonic.size == 0:
        return y
    y_harmonic = librosa.effects.preemphasis(y_harmonic, coef=PREEMPHASIS_COEF)
    peak = float(np.max(np.abs(y_harmonic))) if y_harmonic.size else 0.0
    if peak > 1e-12:
        y_harmonic = y_harmonic / peak
    return y_harmonic


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


def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Safe Pearson r between two 1-D vectors (same as corrcoef[0,1], NaN-safe)."""
    if a.size != b.size or a.size != 12:
        return -1.0
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return -1.0
    r = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(r):
        return -1.0
    return r


def _estimate_key_krumhansl_schmuckler(chroma: np.ndarray) -> tuple[str, float]:
    """
    Krumhansl–Schmuckler: mean chroma (12,) vs rolled major/minor templates; pick best of 24.

    Confidence combines (1) strength of the winning correlation in [0, 1] and
    (2) separation from the runner-up (ambiguous keys get a lower score).
    """
    if chroma.ndim != 2 or chroma.shape[0] != 12:
        return "unknown", 0.0

    chroma_mean = np.mean(chroma, axis=1, dtype=np.float64)
    if not np.any(np.isfinite(chroma_mean)) or float(np.max(np.abs(chroma_mean))) < 1e-12:
        return "unknown", 0.0

    correlations: list[float] = []
    labels: list[str] = []

    for tonic in range(12):
        major_t = np.roll(MAJOR_PROFILE, tonic)
        minor_t = np.roll(MINOR_PROFILE, tonic)
        correlations.append(_pearson_correlation(chroma_mean, major_t))
        labels.append(_major_key_label(tonic))
        correlations.append(_pearson_correlation(chroma_mean, minor_t))
        labels.append(_minor_key_label(tonic))

    corrs = np.asarray(correlations, dtype=np.float64)
    best_idx = int(np.argmax(corrs))
    best_r = float(corrs[best_idx])
    second_r = float(np.partition(corrs, -2)[-2]) if corrs.size >= 2 else best_r
    margin = max(0.0, best_r - second_r)

    # Map best correlation from [-1, 1] to a base score in [0, 1].
    base = float(np.clip((best_r + 1.0) / 2.0, 0.0, 1.0))
    # Sharpen with separation: typical margins are small (0.02–0.15); cap contribution.
    separation = float(np.clip(margin / 0.12, 0.0, 1.0))
    confidence = float(np.clip(0.55 * base + 0.45 * base * (0.35 + 0.65 * separation), 0.0, 1.0))

    return labels[best_idx], round(confidence, 3)


def analyze_audio_bytes(data: bytes, filename: str = "audio") -> AnalysisResult:
    """Load audio from bytes and return BPM + key with pre-processing and beat confidence."""
    y, sr = _load_body_or_fallback(data)
    if y.size == 0:
        return AnalysisResult(bpm=0, bpm_confidence=0.0, key="unknown", confidence=0.0)

    y = _preprocess(y, sr)
    if y.size == 0:
        return AnalysisResult(bpm=0, bpm_confidence=0.0, key="unknown", confidence=0.0)

    # BPM: full-band onset on the same analysis window (post-intro when available).
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    bpm_float, beat_frames = _bpm_from_beat_track(onset_env, sr)
    bpm_rounded = int(round(bpm_float))
    if bpm_rounded < 1:
        bpm_rounded = 1

    bpm_conf = _beat_consistency_score(beat_frames, sr)
    alternates = _half_double_alternates(bpm_rounded)

    y_key = _harmonic_preemphasis(y, sr)
    chroma = librosa.feature.chroma_cqt(
        y=y_key,
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    key_name, key_conf = _estimate_key_krumhansl_schmuckler(chroma)

    return AnalysisResult(
        bpm=bpm_rounded,
        bpm_alternates=alternates,
        bpm_confidence=bpm_conf,
        key=key_name,
        confidence=key_conf,
    )
