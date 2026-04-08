"""OpenAI refinement layer: polish analysis for DJs and producers."""

from __future__ import annotations

import json
import os
from typing import Any

SYSTEM_PROMPT = """Eres un ingeniero de audio profesional, productor musical y DJ con experiencia en mezcla armónica (Camelot), análisis espectral y preparación de pistas para club o estudio.

Recibirás resultados automáticos de un detector (BPM, tonalidad, código Camelot, lista de claves compatibles y confianza).

Instrucciones:
- Responde siempre en español claro y directo.
- No inventes BPM ni tonalidad distintos de los valores de entrada; si comentas incertidumbre, hazlo sin contradecir el JSON.
- Explica brevemente cómo usar el código Camelot y las claves compatibles en una mezcla real.
- Devuelve ÚNICAMENTE un objeto JSON válido que siga el esquema pedido en el mensaje del usuario, sin markdown ni texto fuera del JSON."""

USER_TEMPLATE = """Datos del análisis automático:
- BPM detectado (entero, sin decimales): {bpm}
- Confianza de tempo (estabilidad del beat, 0-1): {bpm_confidence}
- Interpretaciones alternativas (doble tiempo / medio tiempo si aplica): {bpm_alternates_json}
- Tonalidad estimada: {key}
- Código Camelot: {camelot}
- Claves compatibles (mezcla armónica): {compatible}
- Confianza tonal (0-1): {confidence}

Devuelve SOLO un objeto JSON con esta forma (sin markdown, sin comentarios):
{{
  "title": "título corto del resultado",
  "summary": "2-4 frases en español que integren BPM, tonalidad y uso práctico",
  "mixing_tips": ["consejo 1", "consejo 2", "consejo 3"],
  "camelot_note": "una frase sobre el código Camelot y vecinos armónicos",
  "refined_bpm": {bpm},
  "refined_key": "{key}",
  "refined_camelot": "{camelot}",
  "compatible_keys": {compatible_json}
}}"""


def refine_with_openai(
    bpm: int,
    key: str,
    camelot: str,
    compatible_keys: list[str],
    confidence: float,
    bpm_confidence: float,
    bpm_alternates: dict[str, int],
) -> dict[str, Any] | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)
    compatible_json = json.dumps(compatible_keys, ensure_ascii=False)
    alternates_json = json.dumps(bpm_alternates, ensure_ascii=False)
    user = USER_TEMPLATE.format(
        bpm=bpm,
        bpm_confidence=bpm_confidence,
        bpm_alternates_json=alternates_json,
        key=key,
        camelot=camelot,
        compatible=", ".join(compatible_keys),
        confidence=confidence,
        compatible_json=compatible_json,
    )

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.35,
        response_format={"type": "json_object"},
    )
    text = (resp.choices[0].message.content or "").strip()
    return json.loads(text)


def fallback_payload(
    bpm: int,
    key: str,
    camelot: str,
    compatible_keys: list[str],
    confidence: float,
    bpm_confidence: float,
    bpm_alternates: dict[str, int],
    note: str | None = None,
) -> dict[str, Any]:
    alt_note = ""
    if bpm_alternates:
        parts = [f"{k}: {v} BPM" for k, v in bpm_alternates.items()]
        alt_note = " Alternativas habituales: " + "; ".join(parts) + "."
    return {
        "title": "Análisis",
        "summary": (
            f"Tempo aproximado {bpm} BPM (confianza de beat {bpm_confidence:.0%}), "
            f"tonalidad estimada {key} (Camelot {camelot})."
            + alt_note
            + " "
            + (note or "Configura OPENAI_API_KEY para un informe ampliado por IA.")
        ),
        "mixing_tips": [
            "Prueba mezclas con claves adyacentes en la rueda Camelot (+1 / -1 mismo modo).",
            f"Vecinos armónicos sugeridos: {', '.join(compatible_keys)}.",
            "Si el BPM cae en zona ambigua (muy lento o muy rápido), compara con la alternativa de doble/medio tiempo.",
        ],
        "camelot_note": f"En Camelot, {camelot} combina bien con {', '.join(compatible_keys)} en mezcla armónica típica.",
        "refined_bpm": bpm,
        "refined_key": key,
        "refined_camelot": camelot,
        "compatible_keys": compatible_keys,
    }
