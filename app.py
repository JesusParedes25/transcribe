"""
TranscribeYA - Transcriptor de Audio a Texto
=============================================
Servicio web para transcripción de audio largo en español (y otros idiomas)
usando OpenAI Whisper. Soporta archivos grandes con progreso en tiempo real.

Uso:
    pip install -r requirements.txt
    python app.py

Luego abre http://localhost:8000 en tu navegador.
"""

import os
import uuid
import time
import json
import shutil
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

import whisper
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Límites
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
FREE_MINUTES_LIMIT = int(os.getenv("FREE_MINUTES_LIMIT", "30"))  # Minutos gratis
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".wma", ".aac", ".opus", ".webm", ".mp4"}

# Modelo Whisper
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")  # tiny, base, small, medium, large
model = None  # Se carga al iniciar

# Estado de trabajos en memoria (en producción usar Redis/DB)
jobs: dict = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcribeya")


# ---------------------------------------------------------------------------
# Ciclo de vida de la app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Cargando modelo Whisper '{MODEL_SIZE}' en {device}...")
    model = whisper.load_model(MODEL_SIZE, device=device)
    logger.info(f"Modelo cargado exitosamente en {device}")
    yield
    logger.info("Cerrando aplicación...")


app = FastAPI(
    title="TranscribeYA",
    description="Transcripción de audio a texto en español",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def get_audio_duration(filepath: str) -> float:
    """Obtiene la duración del audio en segundos usando ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries",
             "format=duration", "-of", "csv=p=0", filepath],
            capture_output=True, text=True, timeout=30
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def format_timestamp(seconds: float) -> str:
    """Convierte segundos a formato HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_srt_timestamp(seconds: float) -> str:
    """Convierte segundos a formato SRT (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cleanup_old_files(max_age_hours: int = 24):
    """Elimina archivos de más de N horas."""
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        for f in directory.iterdir():
            if f.is_file():
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    f.unlink(missing_ok=True)


def transcribe_audio(job_id: str, filepath: str, language: str, output_format: str):
    """Ejecuta la transcripción de Whisper y actualiza el estado del job."""
    global model
    try:
        jobs[job_id]["status"] = "transcribing"
        jobs[job_id]["message"] = "Transcribiendo audio..."

        # Opciones de transcripción
        transcribe_opts = {
            "verbose": False,
            "word_timestamps": True,
        }
        if language != "auto":
            transcribe_opts["language"] = language

        # Callback de progreso usando los segmentos
        result = model.transcribe(filepath, **transcribe_opts)

        detected_lang = result.get("language", language)
        jobs[job_id]["detected_language"] = detected_lang
        segments = result.get("segments", [])
        total_segments = len(segments)

        # --- Generar salidas ---
        jobs[job_id]["message"] = "Generando archivos de salida..."

        # 1. Texto plano con timestamps
        txt_lines = []
        for seg in segments:
            ts = format_timestamp(seg["start"])
            txt_lines.append(f"[{ts}] {seg['text'].strip()}")
        txt_content = "\n".join(txt_lines)

        # 2. Texto limpio (sin timestamps)
        clean_lines = [seg["text"].strip() for seg in segments]
        clean_content = "\n".join(clean_lines)

        # 3. SRT (subtítulos)
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start_ts = format_srt_timestamp(seg["start"])
            end_ts = format_srt_timestamp(seg["end"])
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_ts} --> {end_ts}")
            srt_lines.append(seg["text"].strip())
            srt_lines.append("")
        srt_content = "\n".join(srt_lines)

        # 4. JSON detallado
        json_data = {
            "metadata": {
                "source_file": jobs[job_id]["original_filename"],
                "duration_seconds": jobs[job_id]["duration"],
                "language": detected_lang,
                "model": MODEL_SIZE,
                "transcribed_at": datetime.now().isoformat(),
            },
            "segments": [
                {
                    "id": i,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for i, seg in enumerate(segments)
            ],
            "full_text": clean_content,
        }

        # Guardar archivos
        base_name = Path(jobs[job_id]["original_filename"]).stem
        outputs = {}

        txt_path = OUTPUT_DIR / f"{job_id}_{base_name}.txt"
        txt_path.write_text(txt_content, encoding="utf-8")
        outputs["txt"] = str(txt_path)

        clean_path = OUTPUT_DIR / f"{job_id}_{base_name}_clean.txt"
        clean_path.write_text(clean_content, encoding="utf-8")
        outputs["clean_txt"] = str(clean_path)

        srt_path = OUTPUT_DIR / f"{job_id}_{base_name}.srt"
        srt_path.write_text(srt_content, encoding="utf-8")
        outputs["srt"] = str(srt_path)

        json_path = OUTPUT_DIR / f"{job_id}_{base_name}.json"
        json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs["json"] = str(json_path)

        # Actualizar job
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Transcripción completada"
        jobs[job_id]["outputs"] = outputs
        jobs[job_id]["preview"] = txt_content[:3000]
        jobs[job_id]["total_segments"] = total_segments
        jobs[job_id]["completed_at"] = datetime.now().isoformat()

        logger.info(f"Job {job_id} completado: {total_segments} segmentos")

    except Exception as e:
        logger.exception(f"Error en job {job_id}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Endpoints API
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_audio(
    file: UploadFile = File(...),
    language: str = Form("es"),
    output_format: str = Form("all"),
):
    """Sube un archivo de audio y comienza la transcripción."""
    # Validar extensión
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Formato no soportado: {ext}. Usa: {', '.join(ALLOWED_EXTENSIONS)}")

    # Generar ID único
    job_id = str(uuid.uuid4())[:8]

    # Guardar archivo
    upload_path = UPLOAD_DIR / f"{job_id}{ext}"
    total_size = 0
    with open(upload_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE_BYTES:
                upload_path.unlink(missing_ok=True)
                raise HTTPException(413, f"Archivo demasiado grande. Máximo: {MAX_FILE_SIZE_MB}MB")
            f.write(chunk)

    # Obtener duración
    duration = get_audio_duration(str(upload_path))
    duration_minutes = duration / 60

    # Registrar job
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "message": "En cola de procesamiento...",
        "original_filename": file.filename,
        "file_path": str(upload_path),
        "file_size_mb": round(total_size / (1024 * 1024), 1),
        "duration": round(duration, 1),
        "duration_minutes": round(duration_minutes, 1),
        "language": language,
        "output_format": output_format,
        "created_at": datetime.now().isoformat(),
        "outputs": {},
        "preview": "",
        "is_free": duration_minutes <= FREE_MINUTES_LIMIT,
    }

    # Lanzar transcripción en background
    import threading
    thread = threading.Thread(
        target=transcribe_audio,
        args=(job_id, str(upload_path), language, output_format),
        daemon=True,
    )
    thread.start()

    return JSONResponse({
        "job_id": job_id,
        "duration_minutes": round(duration_minutes, 1),
        "file_size_mb": round(total_size / (1024 * 1024), 1),
        "is_free": duration_minutes <= FREE_MINUTES_LIMIT,
        "message": "Transcripción iniciada",
    })


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Consulta el estado de un trabajo de transcripción."""
    if job_id not in jobs:
        raise HTTPException(404, "Trabajo no encontrado")
    job = jobs[job_id]
    return JSONResponse({
        "id": job["id"],
        "status": job["status"],
        "message": job["message"],
        "duration_minutes": job.get("duration_minutes", 0),
        "detected_language": job.get("detected_language"),
        "total_segments": job.get("total_segments", 0),
        "preview": job.get("preview", ""),
        "outputs": list(job.get("outputs", {}).keys()),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at"),
    })


@app.get("/api/download/{job_id}/{format}")
async def download_output(job_id: str, format: str):
    """Descarga el resultado en el formato especificado."""
    if job_id not in jobs:
        raise HTTPException(404, "Trabajo no encontrado")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(400, "La transcripción aún no está lista")

    outputs = job.get("outputs", {})
    if format not in outputs:
        raise HTTPException(404, f"Formato '{format}' no disponible. Disponibles: {list(outputs.keys())}")

    filepath = outputs[format]
    filename = Path(filepath).name
    return FileResponse(filepath, filename=filename, media_type="application/octet-stream")


@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "model": MODEL_SIZE,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "active_jobs": sum(1 for j in jobs.values() if j["status"] == "transcribing"),
        "total_jobs": len(jobs),
    }


# ---------------------------------------------------------------------------
# Frontend (HTML embebido)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return FRONTEND_HTML


# ---------------------------------------------------------------------------
# HTML / CSS / JS del frontend
# ---------------------------------------------------------------------------
FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TranscribeYA — Transcripción de Audio Inteligente</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {
    --bg: #0a0a0f;
    --bg-card: #12121a;
    --bg-card-hover: #1a1a26;
    --bg-input: #0e0e16;
    --border: #2a2a3a;
    --border-focus: #6366f1;
    --text: #e4e4eb;
    --text-muted: #8888a0;
    --text-dim: #55556a;
    --accent: #6366f1;
    --accent-glow: rgba(99, 102, 241, 0.3);
    --green: #22c55e;
    --green-glow: rgba(34, 197, 94, 0.2);
    --amber: #f59e0b;
    --red: #ef4444;
    --radius: 12px;
    --radius-sm: 8px;
    --font: 'DM Sans', sans-serif;
    --mono: 'JetBrains Mono', monospace;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.6;
}

/* --- Fondo con gradiente sutil --- */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse at 20% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 100%, rgba(139, 92, 246, 0.06) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

.container {
    max-width: 780px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
    position: relative;
    z-index: 1;
}

/* --- Header --- */
header {
    text-align: center;
    margin-bottom: 3rem;
    padding-top: 1.5rem;
}

header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-bottom: 0.4rem;
    background: linear-gradient(135deg, #e4e4eb 40%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

header p {
    color: var(--text-muted);
    font-size: 1.05rem;
    font-weight: 300;
}

/* --- Cards --- */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: #3a3a50; }

.card-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 1.2rem;
}

/* --- Zona de upload --- */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 3rem 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s;
    position: relative;
    overflow: hidden;
}
.upload-zone:hover, .upload-zone.dragover {
    border-color: var(--accent);
    background: rgba(99, 102, 241, 0.04);
}
.upload-zone.has-file {
    border-color: var(--green);
    border-style: solid;
    background: var(--green-glow);
}
.upload-zone input[type="file"] {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
}

.upload-icon {
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
    opacity: 0.7;
}
.upload-zone p { color: var(--text-muted); font-size: 0.95rem; }
.upload-zone .filename {
    font-family: var(--mono);
    font-size: 0.85rem;
    color: var(--green);
    margin-top: 0.5rem;
    word-break: break-all;
}
.upload-zone .fileinfo {
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-top: 0.25rem;
}

/* --- Opciones --- */
.options-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}

label.option-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    display: block;
    margin-bottom: 0.4rem;
}

select {
    width: 100%;
    padding: 0.65rem 0.9rem;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text);
    font-family: var(--font);
    font-size: 0.9rem;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%238888a0' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 0.8rem center;
    cursor: pointer;
    transition: border-color 0.2s;
}
select:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
}

/* --- Botón principal --- */
.btn-primary {
    width: 100%;
    padding: 0.9rem;
    margin-top: 1.5rem;
    border: none;
    border-radius: var(--radius-sm);
    background: var(--accent);
    color: white;
    font-family: var(--font);
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.01em;
}
.btn-primary:hover:not(:disabled) {
    background: #5558e6;
    box-shadow: 0 4px 24px var(--accent-glow);
    transform: translateY(-1px);
}
.btn-primary:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

/* --- Progreso --- */
#progress-section { display: none; }
#progress-section.active { display: block; }

.progress-status {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 1rem;
}
.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.5s ease-in-out infinite;
}
.status-dot.done { background: var(--green); animation: none; }
.status-dot.error { background: var(--red); animation: none; }
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

.status-text {
    font-size: 0.95rem;
    color: var(--text);
}

.progress-bar-container {
    width: 100%;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 0.5rem;
}
.progress-bar {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.5s ease;
    width: 0%;
}
.progress-bar.indeterminate {
    width: 40%;
    animation: indeterminate 1.5s ease-in-out infinite;
}
@keyframes indeterminate {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(350%); }
}

.progress-meta {
    font-size: 0.78rem;
    color: var(--text-dim);
    font-family: var(--mono);
}

/* --- Resultados --- */
#results-section { display: none; }
#results-section.active { display: block; }

.result-stats {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.stat {
    display: flex;
    flex-direction: column;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: var(--mono);
    color: var(--accent);
}
.stat-label {
    font-size: 0.72rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.preview-box {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.2rem;
    max-height: 350px;
    overflow-y: auto;
    font-family: var(--mono);
    font-size: 0.82rem;
    line-height: 1.8;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
    margin-bottom: 1.5rem;
}
.preview-box::-webkit-scrollbar { width: 6px; }
.preview-box::-webkit-scrollbar-track { background: transparent; }
.preview-box::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.download-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.8rem;
}

.btn-download {
    padding: 0.7rem 1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--bg-input);
    color: var(--text);
    font-family: var(--font);
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    text-align: center;
    text-decoration: none;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
}
.btn-download:hover {
    border-color: var(--accent);
    background: rgba(99, 102, 241, 0.06);
    color: white;
}
.btn-download .dl-icon { font-size: 1rem; }
.btn-download .dl-label { font-size: 0.78rem; color: var(--text-dim); }

/* --- Footer --- */
footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border);
    color: var(--text-dim);
    font-size: 0.8rem;
}
footer a {
    color: var(--accent);
    text-decoration: none;
}
footer a:hover { text-decoration: underline; }

/* --- Tips --- */
.tips {
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-top: 1rem;
    padding: 0.8rem 1rem;
    background: rgba(99, 102, 241, 0.04);
    border-radius: var(--radius-sm);
    border-left: 3px solid var(--accent);
}

/* --- Responsive --- */
@media (max-width: 600px) {
    .container { padding: 1rem; }
    header h1 { font-size: 1.8rem; }
    .card { padding: 1.4rem; }
    .options-row { grid-template-columns: 1fr; }
    .upload-zone { padding: 2rem 1rem; }
    .result-stats { gap: 1rem; }
}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>TranscribeYA</h1>
        <p>Convierte audio a texto con inteligencia artificial</p>
    </header>

    <!-- Upload -->
    <div class="card">
        <div class="card-title">① Sube tu audio</div>
        <div class="upload-zone" id="upload-zone">
            <input type="file" id="file-input" accept=".mp3,.wav,.m4a,.ogg,.flac,.wma,.aac,.opus,.webm,.mp4">
            <div class="upload-icon">🎙️</div>
            <p id="upload-text">Arrastra tu archivo aquí o haz clic para seleccionar</p>
            <div class="filename" id="filename" style="display:none"></div>
            <div class="fileinfo" id="fileinfo" style="display:none"></div>
        </div>

        <div class="options-row">
            <div>
                <label class="option-label">Idioma</label>
                <select id="language">
                    <option value="es" selected>Español</option>
                    <option value="en">English</option>
                    <option value="pt">Português</option>
                    <option value="fr">Français</option>
                    <option value="de">Deutsch</option>
                    <option value="it">Italiano</option>
                    <option value="auto">Auto-detectar</option>
                </select>
            </div>
            <div>
                <label class="option-label">Modelo</label>
                <select id="model-info" disabled>
                    <option>""" + MODEL_SIZE.capitalize() + """ (servidor)</option>
                </select>
            </div>
        </div>

        <button class="btn-primary" id="btn-transcribe" disabled>
            Transcribir Audio
        </button>

        <div class="tips">
            <strong>Formatos soportados:</strong> MP3, WAV, M4A, OGG, FLAC, AAC, OPUS, WEBM, MP4<br>
            <strong>Tamaño máximo:</strong> """ + str(MAX_FILE_SIZE_MB) + """MB &nbsp;·&nbsp;
            <strong>Gratis hasta:</strong> """ + str(FREE_MINUTES_LIMIT) + """ minutos
        </div>
    </div>

    <!-- Progreso -->
    <div class="card" id="progress-section">
        <div class="card-title">② Procesando</div>
        <div class="progress-status">
            <div class="status-dot" id="status-dot"></div>
            <span class="status-text" id="status-text">Subiendo archivo...</span>
        </div>
        <div class="progress-bar-container">
            <div class="progress-bar indeterminate" id="progress-bar"></div>
        </div>
        <div class="progress-meta" id="progress-meta"></div>
    </div>

    <!-- Resultados -->
    <div class="card" id="results-section">
        <div class="card-title">③ Resultado</div>
        <div class="result-stats" id="result-stats"></div>
        <div class="preview-box" id="preview-box"></div>
        <div class="download-grid" id="download-grid"></div>
    </div>

    <footer>
        Hecho con Whisper (OpenAI) + FastAPI &nbsp;·&nbsp;
        <a href="/api/health" target="_blank">Estado del servidor</a>
        <br><br>
        ¿Te fue útil? &nbsp;
        <a href="#" id="donate-link">☕ Invítame un café</a>
    </footer>
</div>

<script>
const $ = id => document.getElementById(id);
const zone = $('upload-zone');
const fileInput = $('file-input');
const btn = $('btn-transcribe');
let selectedFile = null;
let currentJobId = null;

// --- Drag & Drop ---
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', e => {
    if (e.target.files.length) handleFileSelect(e.target.files[0]);
});

function handleFileSelect(file) {
    selectedFile = file;
    zone.classList.add('has-file');
    $('upload-text').textContent = '✓ Archivo seleccionado';
    $('filename').textContent = file.name;
    $('filename').style.display = 'block';
    const sizeMB = (file.size / 1024 / 1024).toFixed(1);
    $('fileinfo').textContent = sizeMB + ' MB';
    $('fileinfo').style.display = 'block';
    btn.disabled = false;
}

// --- Transcribir ---
btn.addEventListener('click', async () => {
    if (!selectedFile) return;

    btn.disabled = true;
    btn.textContent = 'Subiendo...';
    $('progress-section').classList.add('active');
    $('results-section').classList.remove('active');
    $('status-text').textContent = 'Subiendo archivo al servidor...';
    $('progress-meta').textContent = '';

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('language', $('language').value);

    try {
        const resp = await fetch('/api/upload', { method: 'POST', body: formData });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Error al subir');
        }
        const data = await resp.json();
        currentJobId = data.job_id;

        $('status-text').textContent = 'Transcribiendo audio...';
        $('progress-meta').textContent =
            `Duración: ${data.duration_minutes} min · ${data.file_size_mb} MB`;

        pollStatus();
    } catch (err) {
        $('status-text').textContent = '❌ ' + err.message;
        $('status-dot').classList.add('error');
        $('progress-bar').classList.remove('indeterminate');
        btn.disabled = false;
        btn.textContent = 'Transcribir Audio';
    }
});

async function pollStatus() {
    if (!currentJobId) return;
    try {
        const resp = await fetch(`/api/status/${currentJobId}`);
        const data = await resp.json();

        $('status-text').textContent = data.message;

        if (data.status === 'completed') {
            $('status-dot').classList.add('done');
            $('progress-bar').classList.remove('indeterminate');
            $('progress-bar').style.width = '100%';
            showResults(data);
            btn.textContent = 'Transcribir Audio';
            return;
        }

        if (data.status === 'error') {
            $('status-dot').classList.add('error');
            $('progress-bar').classList.remove('indeterminate');
            btn.disabled = false;
            btn.textContent = 'Transcribir Audio';
            return;
        }

        setTimeout(pollStatus, 2000);
    } catch {
        setTimeout(pollStatus, 3000);
    }
}

function showResults(data) {
    $('results-section').classList.add('active');

    // Stats
    $('result-stats').innerHTML = `
        <div class="stat">
            <span class="stat-value">${data.duration_minutes || '—'}</span>
            <span class="stat-label">Minutos</span>
        </div>
        <div class="stat">
            <span class="stat-value">${data.total_segments || '—'}</span>
            <span class="stat-label">Segmentos</span>
        </div>
        <div class="stat">
            <span class="stat-value">${data.detected_language?.toUpperCase() || '—'}</span>
            <span class="stat-label">Idioma</span>
        </div>
    `;

    // Preview
    $('preview-box').textContent = data.preview || 'Sin vista previa disponible';

    // Downloads
    const formats = {
        txt: { icon: '📄', label: 'Texto + Timestamps' },
        clean_txt: { icon: '📝', label: 'Texto Limpio' },
        srt: { icon: '🎬', label: 'Subtítulos SRT' },
        json: { icon: '📊', label: 'JSON Detallado' },
    };

    let html = '';
    for (const fmt of (data.outputs || [])) {
        const info = formats[fmt] || { icon: '📎', label: fmt };
        html += `
            <a class="btn-download" href="/api/download/${currentJobId}/${fmt}" download>
                <span class="dl-icon">${info.icon}</span>
                <span>${info.label}</span>
            </a>
        `;
    }
    $('download-grid').innerHTML = html;
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    cleanup_old_files()  # Limpiar al iniciar
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Iniciando TranscribeYA en http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
