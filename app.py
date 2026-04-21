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
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "2048"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
FREE_MINUTES_LIMIT = int(os.getenv("FREE_MINUTES_LIMIT", "150"))  # 2h30 gratis
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".wma", ".aac", ".opus", ".webm", ".mp4"}

# Modelos Whisper disponibles
AVAILABLE_MODELS = ["tiny", "base", "small", "medium"]
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "small")
models = {}  # Cache de modelos cargados
device = None  # Se establece al iniciar

# Estado de trabajos en memoria (en producción usar Redis/DB)
jobs: dict = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcribeya")


# ---------------------------------------------------------------------------
# Ciclo de vida de la app
# ---------------------------------------------------------------------------
def get_model(model_name: str):
    """Obtiene un modelo de la cache o lo carga si no existe."""
    global models, device
    if model_name not in AVAILABLE_MODELS:
        model_name = DEFAULT_MODEL
    
    if model_name not in models:
        logger.info(f"Cargando modelo Whisper '{model_name}' en {device}...")
        models[model_name] = whisper.load_model(model_name, device=device)
        logger.info(f"Modelo '{model_name}' cargado exitosamente")
    
    return models[model_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Dispositivo de transcripción: {device}")
    # Pre-cargar el modelo por defecto
    get_model(DEFAULT_MODEL)
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


def transcribe_audio(job_id: str, filepath: str, language: str, output_format: str, model_name: str):
    """Ejecuta la transcripción de Whisper y actualiza el estado del job."""
    try:
        jobs[job_id]["status"] = "transcribing"
        jobs[job_id]["progress"] = 0
        jobs[job_id]["message"] = f"Cargando modelo {model_name}..."
        
        # Obtener el modelo (se carga si no está en cache)
        model = get_model(model_name)
        jobs[job_id]["message"] = "Transcribiendo audio..."
        
        # Iniciar hilo de progreso estimado
        duration = jobs[job_id].get("duration", 0)
        import threading
        stop_progress = threading.Event()
        
        def update_progress():
            """Actualiza el progreso estimado basado en velocidad del modelo."""
            # Velocidades aproximadas (segundos de audio por segundo real)
            speed_factors = {"tiny": 32, "base": 16, "small": 6, "medium": 2}
            speed = speed_factors.get(model_name, 6)
            estimated_time = duration / speed if speed > 0 else duration
            start_time = time.time()
            
            while not stop_progress.is_set():
                elapsed = time.time() - start_time
                progress = min(95, int((elapsed / estimated_time) * 100)) if estimated_time > 0 else 0
                jobs[job_id]["progress"] = progress
                if progress < 95:
                    time.sleep(0.5)
                else:
                    break
        
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()

        # Opciones de transcripción
        transcribe_opts = {
            "verbose": False,
            "word_timestamps": True,
        }
        if language != "auto":
            transcribe_opts["language"] = language

        result = model.transcribe(filepath, **transcribe_opts)
        
        # Detener hilo de progreso
        stop_progress.set()

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
                "model": model_name,
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
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Transcripción completada"
        jobs[job_id]["outputs"] = outputs
        jobs[job_id]["preview"] = txt_content[:3000]
        jobs[job_id]["total_segments"] = total_segments
        jobs[job_id]["completed_at"] = datetime.now().isoformat()

        logger.info(f"Job {job_id} completado: {total_segments} segmentos")
        
        # Eliminar archivo de audio para ahorrar espacio
        try:
            Path(filepath).unlink(missing_ok=True)
            logger.info(f"Audio eliminado: {filepath}")
        except Exception as del_err:
            logger.warning(f"No se pudo eliminar audio {filepath}: {del_err}")

    except Exception as e:
        logger.exception(f"Error en job {job_id}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = f"Error: {str(e)}"
        # También eliminar el audio en caso de error
        try:
            Path(filepath).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Endpoints API
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_audio(
    file: UploadFile = File(...),
    language: str = Form("es"),
    output_format: str = Form("all"),
    model: str = Form("small"),
):
    """Sube un archivo de audio y comienza la transcripción."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Formato no soportado: {ext}. Usa: {', '.join(ALLOWED_EXTENSIONS)}")

    job_id = str(uuid.uuid4())[:8]

    upload_path = UPLOAD_DIR / f"{job_id}{ext}"
    total_size = 0
    with open(upload_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE_BYTES:
                upload_path.unlink(missing_ok=True)
                raise HTTPException(413, f"Archivo demasiado grande. Máximo: {MAX_FILE_SIZE_MB}MB")
            f.write(chunk)

    duration = get_audio_duration(str(upload_path))
    duration_minutes = duration / 60

    # Validar modelo seleccionado
    selected_model = model if model in AVAILABLE_MODELS else DEFAULT_MODEL
    
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "message": "En cola de procesamiento...",
        "progress": 0,
        "original_filename": file.filename,
        "file_path": str(upload_path),
        "file_size_mb": round(total_size / (1024 * 1024), 1),
        "duration": round(duration, 1),
        "duration_minutes": round(duration_minutes, 1),
        "language": language,
        "output_format": output_format,
        "model": selected_model,
        "created_at": datetime.now().isoformat(),
        "outputs": {},
        "preview": "",
        "is_free": duration_minutes <= FREE_MINUTES_LIMIT,
    }

    import threading
    thread = threading.Thread(
        target=transcribe_audio,
        args=(job_id, str(upload_path), language, output_format, selected_model),
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
        "progress": job.get("progress", 0),
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
        "default_model": DEFAULT_MODEL,
        "available_models": AVAILABLE_MODELS,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "active_jobs": sum(1 for j in jobs.values() if j["status"] == "transcribing"),
        "total_jobs": len(jobs),
    }


# ---------------------------------------------------------------------------
# Frontend (HTML embebido) — Emojis como entidades HTML para evitar surrogates
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    html = _build_frontend_html()
    return HTMLResponse(content=html)


def _build_frontend_html() -> str:
    max_size = str(MAX_FILE_SIZE_MB)
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TranscribeYA &#8212; Transcripci&#243;n de Audio Inteligente</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {{
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
    --amber-glow: rgba(245, 158, 11, 0.15);
    --red: #ef4444;
    --radius: 12px;
    --radius-sm: 8px;
    --font: 'DM Sans', sans-serif;
    --mono: 'JetBrains Mono', monospace;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.6;
}}

body::before {{
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse at 20% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 100%, rgba(139, 92, 246, 0.06) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}}

.container {{
    max-width: 780px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
    position: relative;
    z-index: 1;
}}

header {{
    text-align: center;
    margin-bottom: 3rem;
    padding-top: 1.5rem;
}}

header h1 {{
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-bottom: 0.4rem;
    background: linear-gradient(135deg, #e4e4eb 40%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

header p {{
    color: var(--text-muted);
    font-size: 1.05rem;
    font-weight: 300;
}}

.card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.2s;
}}
.card:hover {{ border-color: #3a3a50; }}

.card-title {{
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 1.2rem;
}}

.upload-zone {{
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 3rem 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s;
    position: relative;
    overflow: hidden;
}}
.upload-zone:hover, .upload-zone.dragover {{
    border-color: var(--accent);
    background: rgba(99, 102, 241, 0.04);
}}
.upload-zone.has-file {{
    border-color: var(--green);
    border-style: solid;
    background: var(--green-glow);
}}
.upload-zone input[type="file"] {{
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
}}

.upload-icon {{
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
    opacity: 0.7;
}}
.upload-zone p {{ color: var(--text-muted); font-size: 0.95rem; }}
.upload-zone .filename {{
    font-family: var(--mono);
    font-size: 0.85rem;
    color: var(--green);
    margin-top: 0.5rem;
    word-break: break-all;
}}
.upload-zone .fileinfo {{
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-top: 0.25rem;
}}

.options-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}}

label.option-label {{
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    display: block;
    margin-bottom: 0.4rem;
}}

select {{
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
}}
select:focus {{
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
}}

.btn-primary {{
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
}}
.btn-primary:hover:not(:disabled) {{
    background: #5558e6;
    box-shadow: 0 4px 24px var(--accent-glow);
    transform: translateY(-1px);
}}
.btn-primary:disabled {{
    opacity: 0.4;
    cursor: not-allowed;
}}

#progress-section {{ display: none; }}
#progress-section.active {{ display: block; }}

.progress-status {{
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 1rem;
}}
.status-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.5s ease-in-out infinite;
    flex-shrink: 0;
}}
.status-dot.done {{ background: var(--green); animation: none; }}
.status-dot.error {{ background: var(--red); animation: none; }}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.4; transform: scale(0.8); }}
}}

.status-text {{
    font-size: 0.95rem;
    color: var(--text);
}}

.progress-bar-container {{
    width: 100%;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 0.5rem;
}}
.progress-bar {{
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.5s ease;
    width: 0%;
}}
.progress-bar.indeterminate {{
    width: 40%;
    animation: indeterminate 1.5s ease-in-out infinite;
}}
@keyframes indeterminate {{
    0% {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(350%); }}
}}

.progress-meta {{
    font-size: 0.78rem;
    color: var(--text-dim);
    font-family: var(--mono);
}}

#results-section {{ display: none; }}
#results-section.active {{ display: block; }}

.result-stats {{
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}}
.stat {{
    display: flex;
    flex-direction: column;
}}
.stat-value {{
    font-size: 1.4rem;
    font-weight: 700;
    font-family: var(--mono);
    color: var(--accent);
}}
.stat-label {{
    font-size: 0.72rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.preview-box {{
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
}}
.preview-box::-webkit-scrollbar {{ width: 6px; }}
.preview-box::-webkit-scrollbar-track {{ background: transparent; }}
.preview-box::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

.download-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.8rem;
}}

.btn-download {{
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
}}
.btn-download:hover {{
    border-color: var(--accent);
    background: rgba(99, 102, 241, 0.06);
    color: white;
}}

.donate-banner {{
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(251, 191, 36, 0.04) 100%);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-radius: var(--radius);
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    display: none;
}}
.donate-banner.active {{ display: block; }}

.donate-banner .donate-emoji {{
    font-size: 2rem;
    margin-bottom: 0.5rem;
}}
.donate-banner .donate-title {{
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.3rem;
}}
.donate-banner .donate-desc {{
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-bottom: 1rem;
    line-height: 1.5;
}}
.btn-donate {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.7rem 1.8rem;
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: #fff;
    font-family: var(--font);
    font-size: 0.95rem;
    font-weight: 600;
    border: none;
    border-radius: var(--radius-sm);
    cursor: pointer;
    text-decoration: none;
    transition: all 0.25s;
    box-shadow: 0 2px 12px rgba(245, 158, 11, 0.2);
}}
.btn-donate:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(245, 158, 11, 0.35);
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    color: #fff;
}}

.tips {{
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-top: 1rem;
    padding: 0.8rem 1rem;
    background: rgba(99, 102, 241, 0.04);
    border-radius: var(--radius-sm);
    border-left: 3px solid var(--accent);
    line-height: 1.7;
}}

.model-hint {{
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    margin-top: 1rem;
    padding: 0.7rem 1rem;
    background: rgba(34, 197, 94, 0.06);
    border-radius: var(--radius-sm);
    border-left: 3px solid var(--green);
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.5;
}}
.model-hint .hint-icon {{
    flex-shrink: 0;
}}

.progress-percent {{
    font-family: var(--mono);
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--accent);
    margin-left: auto;
}}

footer {{
    text-align: center;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border);
    color: var(--text-dim);
    font-size: 0.8rem;
}}
footer a {{
    color: var(--accent);
    text-decoration: none;
}}
footer a:hover {{ text-decoration: underline; }}

@media (max-width: 600px) {{
    .container {{ padding: 1rem; }}
    header h1 {{ font-size: 1.8rem; }}
    .card {{ padding: 1.4rem; }}
    .options-row {{ grid-template-columns: 1fr; }}
    .upload-zone {{ padding: 2rem 1rem; }}
    .result-stats {{ gap: 1rem; }}
    .donate-banner {{ padding: 1.2rem; }}
}}
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
        <div class="card-title">&#9312; Sube tu audio</div>
        <div class="upload-zone" id="upload-zone">
            <input type="file" id="file-input" accept=".mp3,.wav,.m4a,.ogg,.flac,.wma,.aac,.opus,.webm,.mp4">
            <div class="upload-icon">&#127897;&#65039;</div>
            <p id="upload-text">Arrastra tu archivo aqu&#237; o haz clic para seleccionar</p>
            <div class="filename" id="filename" style="display:none"></div>
            <div class="fileinfo" id="fileinfo" style="display:none"></div>
        </div>

        <div class="options-row">
            <div>
                <label class="option-label">Idioma</label>
                <select id="language">
                    <option value="es" selected>Espa&#241;ol</option>
                    <option value="en">English</option>
                    <option value="pt">Portugu&#234;s</option>
                    <option value="fr">Fran&#231;ais</option>
                    <option value="de">Deutsch</option>
                    <option value="it">Italiano</option>
                    <option value="auto">Auto-detectar</option>
                </select>
            </div>
            <div>
                <label class="option-label">Modelo</label>
                <select id="model-select">
                    <option value="tiny">&#9889; R&#225;pido - Ideal para pruebas r&#225;pidas</option>
                    <option value="base">&#128640; Balanceado - Buena velocidad, calidad aceptable</option>
                    <option value="small" selected>&#11088; Recomendado - Mejor equilibrio calidad/velocidad</option>
                    <option value="medium">&#127942; Alta calidad - M&#225;s preciso, tarda m&#225;s</option>
                </select>
            </div>
        </div>
        
        <div class="model-hint" id="model-hint">
            <span class="hint-icon">&#128161;</span>
            <span id="hint-text">Equilibrio ideal entre velocidad y precisi&#243;n. Funciona bien con la mayor&#237;a de audios.</span>
        </div>

        <button class="btn-primary" id="btn-transcribe" disabled>
            Transcribir Audio
        </button>

        <div class="tips">
            <strong>Formatos:</strong> MP3, WAV, M4A, OGG, FLAC, AAC, OPUS, WEBM, MP4<br>
            <strong>Tama&#241;o m&#225;ximo:</strong> 2GB &#183;
            <strong>100% gratuito</strong> &#183;
            <strong>Totalmente an&#243;nimo</strong> (no guardamos tus audios)
        </div>
    </div>

    <!-- Progress -->
    <div class="card" id="progress-section">
        <div class="card-title">&#9313; Procesando</div>
        <div class="progress-status">
            <div class="status-dot" id="status-dot"></div>
            <span class="status-text" id="status-text">Subiendo archivo...</span>
            <span class="progress-percent" id="progress-percent"></span>
        </div>
        <div class="progress-bar-container">
            <div class="progress-bar" id="progress-bar"></div>
        </div>
        <div class="progress-meta" id="progress-meta"></div>
    </div>

    <!-- Donation banner -->
    <div class="donate-banner" id="donate-banner">
        <div class="donate-emoji">&#9749;</div>
        <div class="donate-title">&#191;Te fue &#250;til esta transcripci&#243;n?</div>
        <div class="donate-desc">
            TranscribeYA es gratuito y sin registro. Si te ahorr&#243; tiempo,
            considera apoyar el proyecto para mantener el servidor activo.
        </div>
        <a class="btn-donate" href="https://link.mercadopago.com.mx/geobint" target="_blank" rel="noopener">
            &#9749; Inv&#237;tame un caf&#233;
        </a>
    </div>

    <!-- Results -->
    <div class="card" id="results-section">
        <div class="card-title">&#9314; Resultado</div>
        <div class="result-stats" id="result-stats"></div>
        <div class="preview-box" id="preview-box"></div>
        <div class="download-grid" id="download-grid"></div>
    </div>

    <footer>
        <a href="https://link.mercadopago.com.mx/geobint" target="_blank" rel="noopener">&#9749; Apoyar el proyecto</a>
    </footer>
</div>

<script>
const $ = id => document.getElementById(id);
const zone = $('upload-zone');
const fileInput = $('file-input');
const btn = $('btn-transcribe');
let selectedFile = null;
let currentJobId = null;

zone.addEventListener('dragover', e => {{ e.preventDefault(); zone.classList.add('dragover'); }});
zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
zone.addEventListener('drop', e => {{
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {{
        fileInput.files = e.dataTransfer.files;
        handleFileSelect(e.dataTransfer.files[0]);
    }}
}});

fileInput.addEventListener('change', e => {{
    if (e.target.files.length) handleFileSelect(e.target.files[0]);
}});

function handleFileSelect(file) {{
    selectedFile = file;
    zone.classList.add('has-file');
    $('upload-text').textContent = '\\u2713 Archivo seleccionado';
    $('filename').textContent = file.name;
    $('filename').style.display = 'block';
    const sizeMB = (file.size / 1024 / 1024).toFixed(1);
    $('fileinfo').textContent = sizeMB + ' MB';
    $('fileinfo').style.display = 'block';
    btn.disabled = false;
}}

// Descripciones amigables para cada modelo
const modelHints = {{
    tiny: 'El m\u00e1s r\u00e1pido. Ideal para probar o audios muy claros. Puede tener errores en palabras dif\u00edciles.',
    base: 'Buen balance. M\u00e1s r\u00e1pido que Small pero menos preciso. Funciona bien con audio claro.',
    small: 'Equilibrio ideal entre velocidad y precisi\u00f3n. Funciona bien con la mayor\u00eda de audios.',
    medium: 'Mayor precisi\u00f3n, especialmente con acentos o ruido de fondo. Tarda m\u00e1s en procesar.'
}};

$('model-select').addEventListener('change', function() {{
    $('hint-text').textContent = modelHints[this.value] || '';
}});

btn.addEventListener('click', async () => {{
    if (!selectedFile) return;

    btn.disabled = true;
    btn.textContent = 'Subiendo...';
    $('progress-section').classList.add('active');
    $('results-section').classList.remove('active');
    $('donate-banner').classList.remove('active');
    $('status-dot').className = 'status-dot';
    $('status-text').textContent = 'Subiendo archivo al servidor...';
    $('progress-meta').textContent = '';
    $('progress-percent').textContent = '';
    $('progress-bar').className = 'progress-bar';
    $('progress-bar').style.width = '0%';

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('language', $('language').value);
    formData.append('model', $('model-select').value);

    try {{
        const resp = await fetch('/api/upload', {{ method: 'POST', body: formData }});
        if (!resp.ok) {{
            const err = await resp.json();
            throw new Error(err.detail || 'Error al subir');
        }}
        const data = await resp.json();
        currentJobId = data.job_id;

        localStorage.setItem('transcribeya_job', JSON.stringify({{
            id: currentJobId,
            filename: selectedFile.name,
            timestamp: Date.now()
        }}));

        $('status-text').textContent = 'Transcribiendo audio... esto puede tardar varios minutos';
        $('progress-meta').textContent =
            'Duraci\\u00f3n: ' + data.duration_minutes + ' min \\u00b7 ' + data.file_size_mb + ' MB';

        pollStatus();
    }} catch (err) {{
        $('status-text').textContent = '\\u274c ' + err.message;
        $('status-dot').classList.add('error');
        $('progress-bar').classList.remove('indeterminate');
        btn.disabled = false;
        btn.textContent = 'Transcribir Audio';
    }}
}});

async function pollStatus() {{
    if (!currentJobId) return;
    try {{
        const resp = await fetch('/api/status/' + currentJobId);
        const data = await resp.json();

        $('status-text').textContent = data.message;
        
        // Actualizar barra de progreso
        var progress = data.progress || 0;
        $('progress-bar').style.width = progress + '%';
        $('progress-percent').textContent = progress + '%';

        if (data.status === 'completed') {{
            $('status-dot').classList.add('done');
            $('progress-bar').style.width = '100%';
            $('progress-percent').textContent = '100%';
            showResults(data);
            btn.textContent = 'Transcribir Audio';
            localStorage.removeItem('transcribeya_job');
            return;
        }}

        if (data.status === 'error') {{
            $('status-dot').classList.add('error');
            $('progress-bar').classList.remove('indeterminate');
            btn.disabled = false;
            btn.textContent = 'Transcribir Audio';
            localStorage.removeItem('transcribeya_job');
            return;
        }}

        setTimeout(pollStatus, 2000);
    }} catch {{
        setTimeout(pollStatus, 3000);
    }}
}}

function showResults(data) {{
    $('results-section').classList.add('active');
    $('donate-banner').classList.add('active');

    $('result-stats').innerHTML =
        '<div class="stat">' +
            '<span class="stat-value">' + (data.duration_minutes || '\\u2014') + '</span>' +
            '<span class="stat-label">Minutos</span>' +
        '</div>' +
        '<div class="stat">' +
            '<span class="stat-value">' + (data.total_segments || '\\u2014') + '</span>' +
            '<span class="stat-label">Segmentos</span>' +
        '</div>' +
        '<div class="stat">' +
            '<span class="stat-value">' + ((data.detected_language || '').toUpperCase() || '\\u2014') + '</span>' +
            '<span class="stat-label">Idioma</span>' +
        '</div>';

    $('preview-box').textContent = data.preview || 'Sin vista previa disponible';

    var formats = {{
        txt:       {{ icon: '&#128196;', label: 'Texto + Timestamps' }},
        clean_txt: {{ icon: '&#128221;', label: 'Texto Limpio' }},
        srt:       {{ icon: '&#127916;', label: 'Subt\\u00edtulos SRT' }},
        json:      {{ icon: '&#128202;', label: 'JSON Detallado' }}
    }};

    var html = '';
    var outputs = data.outputs || [];
    for (var k = 0; k < outputs.length; k++) {{
        var fmt = outputs[k];
        var info = formats[fmt] || {{ icon: '&#128206;', label: fmt }};
        html +=
            '<a class="btn-download" href="/api/download/' + currentJobId + '/' + fmt + '" download>' +
                '<span class="dl-icon">' + info.icon + '</span>' +
                '<span>' + info.label + '</span>' +
            '</a>';
    }}
    $('download-grid').innerHTML = html;
}}

(function recoverJob() {{
    try {{
        var saved = JSON.parse(localStorage.getItem('transcribeya_job'));
        if (saved && saved.id && (Date.now() - saved.timestamp < 3600000)) {{
            currentJobId = saved.id;
            $('progress-section').classList.add('active');
            $('status-text').textContent = 'Reconectando con transcripci\\u00f3n en curso...';
            $('progress-meta').textContent = saved.filename || '';
            btn.disabled = true;
            btn.textContent = 'Procesando...';
            pollStatus();
        }}
    }} catch(e) {{}}
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    cleanup_old_files()
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Iniciando TranscribeYA en http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)