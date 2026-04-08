@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creando entorno virtual...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
echo.
echo Servidor: http://127.0.0.1:5000  ^(Live Server front suele usar puerto 5500^)
echo Coloca OPENAI_API_KEY en backend\.env para informe IA completo.
echo.
uvicorn app.main:app --reload --host 127.0.0.1 --port 5000
