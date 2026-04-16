"""Find The Key — FastAPI application."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

_backend_dir = Path(__file__).resolve().parent.parent
load_dotenv(_backend_dir / ".env")
load_dotenv(_backend_dir.parent / ".env")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.services.ai_refinement import fallback_payload, refine_with_openai
from app.services.audio_analysis import analyze_audio_bytes
from app.services.camelot import camelot_compatible, key_to_camelot
ALLOWED_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".webm"}

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )
logger = logging.getLogger(__name__)


def _int_bpm(value) -> int:
    if value is None:
        return 0
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


app = FastAPI(title="Find The Key", version="1.0.0")

# CORS: en Flask sería «from flask_cors import CORS» y «CORS(app)» justo después de crear app.
# Aquí app es FastAPI, no Flask; no uses flask_cors en este archivo. El equivalente es CORSMiddleware.
# CORS_ORIGINS: lista separada por comas. Además, allow_origin_regex cubre cualquier puerto en local.
_origins = os.getenv(
    "CORS_ORIGINS",
    "http://127.0.0.1:5500,http://localhost:5500,"
    "http://127.0.0.1:5501,http://localhost:5501,"
    "http://127.0.0.1:8080,http://localhost:8080,"
    "http://localhost:3000,http://127.0.0.1:3000,"
    "https://findthekey.es"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost|.*\.onrender\.com|findthekey\.es)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "openai": bool(os.getenv("OPENAI_API_KEY", "").strip()),
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename")
    suf = Path(file.filename).suffix.lower()
    if suf not in ALLOWED_EXT:
        raise HTTPException(
            400,
            f"Unsupported format: {suf}. Allowed: {', '.join(sorted(ALLOWED_EXT))}",
        )
    try:
        data = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Read error: {e}") from e
    if len(data) < 1024:
        raise HTTPException(400, "File too small or empty")

    try:
        result = analyze_audio_bytes(data, file.filename)
    except Exception as e:
        logger.exception("Librosa / analyze_audio_bytes failed for %s", file.filename)
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo analizar el audio: {e}",
        ) from e
    camelot = key_to_camelot(result.key) or "—"
    compatible = camelot_compatible(camelot) if camelot != "—" else []

    ai = refine_with_openai(
        result.bpm,
        result.key,
        camelot,
        compatible,
        result.confidence,
        result.bpm_confidence,
        result.bpm_alternates,
    )
    if ai is None:
        ai = fallback_payload(
            result.bpm,
            result.key,
            camelot,
            compatible,
            result.confidence,
            result.bpm_confidence,
            result.bpm_alternates,
        )
    else:
        # Ensure compatible_keys present if model omitted
        if "compatible_keys" not in ai or not ai["compatible_keys"]:
            ai["compatible_keys"] = compatible

    body = {
        "bpm": _int_bpm(ai.get("refined_bpm", result.bpm)),
        "bpm_alternates": result.bpm_alternates,
        "bpm_confidence": result.bpm_confidence,
        "key": ai.get("refined_key", result.key),
        "camelot": ai.get("refined_camelot", camelot),
        "compatible_keys": ai.get("compatible_keys", compatible),
        "confidence": result.confidence,
        "ai": {
            "title": ai.get("title"),
            "summary": ai.get("summary"),
            "mixing_tips": ai.get("mixing_tips"),
            "camelot_note": ai.get("camelot_note"),
        },
        "analysis_raw": {
            "bpm": result.bpm,
            "bpm_alternates": result.bpm_alternates,
            "bpm_confidence": result.bpm_confidence,
            "key": result.key,
            "camelot": camelot,
        },
    }
    return JSONResponse(content=body)


# Optional: serve frontend from backend (production-style)
_frontend = Path(__file__).resolve().parent.parent.parent / "frontend"
if _frontend.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
