# ===========================================================
# TranscribeYA - Dockerfile
# ===========================================================
# Build:  docker build -t transcribeya .
# Run:    docker run -p 8000:8000 --gpus all transcribeya
# (sin GPU: docker run -p 8000:8000 transcribeya)
# ===========================================================

FROM python:3.11-slim

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descargar modelo (medium por defecto, cambiar si necesitas otro)
ARG WHISPER_MODEL=small
RUN python -c "import whisper; whisper.load_model('${WHISPER_MODEL}')"

# Copiar código
COPY app.py .

# Crear directorios
RUN mkdir -p uploads outputs

# Variables de entorno
ENV WHISPER_MODEL=${WHISPER_MODEL}
ENV PORT=8000
ENV MAX_FILE_SIZE_MB=500
ENV FREE_MINUTES_LIMIT=30

EXPOSE 8000

CMD ["python", "app.py"]
