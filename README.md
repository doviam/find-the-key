# Find The Key

Aplicación web full-stack para **subir un audio** y obtener **BPM**, **tonalidad**, **código Camelot**, **claves compatibles** para mezcla armónica y un **informe refinado** (opcional) vía OpenAI.

## Arquitectura

- **`frontend/`** — HTML, CSS y JavaScript (tema oscuro estilo cabina DJ). Sin framework obligatorio.
- **`backend/`** — Python **FastAPI**: análisis con **librosa** (tempo + Krumhansl–Schmuckler para tonalidad), mapeo Camelot, compatibilidad armónica, y capa **OpenAI** que formatea y mejora el texto de salida.

## Requisitos

- **Python 3.10+**
- Navegador moderno
- **FFmpeg** instalado y en el `PATH` (recomendado para **MP3** y otros formatos comprimidos; WAV/FLAC suelen funcionar sin FFmpeg según tu entorno)

### FFmpeg en Windows

Instálalo desde [ffmpeg.org](https://ffmpeg.org/download.html) o con `winget install FFmpeg`, y comprueba que `ffmpeg -version` funciona en una terminal nueva.

## Puesta en marcha (recomendado: un solo servidor)

Desde la raíz del proyecto:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Crea `backend/.env` (puedes partir de `../.env.example`) y añade tu clave si quieres la capa IA:

```
OPENAI_API_KEY=sk-tu-clave
```

Arranca el backend (sirve la API **y** los archivos estáticos del frontend):

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Abre en el navegador: **http://127.0.0.1:8000**

- API: `POST /api/analyze` (multipart, campo `file`)
- Salud: `GET /api/health`

## Modo separado (frontend en otro puerto)

1. Ejecuta FastAPI como arriba en el puerto **8000**.
2. Sirve `frontend/` con tu herramienta favorita (p. ej. extensión Live Server en el puerto **5500**).
3. Añade en `frontend/index.html` la URL del API en la meta:

   ```html
   <meta name="api-base" content="http://127.0.0.1:8000" />
   ```

4. Ajusta `CORS_ORIGINS` en `.env` para incluir el origen del frontend (por defecto ya incluye varios puertos locales).

## Variables de entorno

| Variable | Descripción |
|----------|-------------|
| `OPENAI_API_KEY` | Si está vacía, el informe IA usa un texto de respaldo fijo en español. |
| `OPENAI_MODEL` | Modelo de chat (por defecto `gpt-4o-mini`). |
| `CORS_ORIGINS` | Lista separada por comas de orígenes permitidos. |

## Respuesta JSON (`/api/analyze`)

Incluye `bpm` (entero), `bpm_alternates` (p. ej. `double_time` / `half_time` si el BPM está en zona ambigua), `bpm_confidence` (0–1, estabilidad del beat), `key`, `camelot`, `compatible_keys`, `confidence` (tonalidad), bloque `ai` y `analysis_raw` con los valores del detector.

## Notas

- BPM y tonalidad son **estimaciones**; canciones con cambios de tempo o modulaciones pueden dar resultados menos estables.
- El prompt de sistema para OpenAI está en `backend/app/services/ai_refinement.py`; puedes sustituirlo por el texto largo que quieras usar como “prompt maestro”.
