# TranscribeYA 🎙️

**Transcriptor de audio a texto** usando OpenAI Whisper. Interfaz web moderna, soporte para archivos largos, múltiples formatos de salida.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Whisper](https://img.shields.io/badge/Whisper-large-orange)

---

## Características

- **Transcripción precisa** con modelos Whisper (tiny → large)
- **Soporte multiidioma**: español, inglés, portugués, francés, alemán, italiano + autodetección
- **Archivos largos**: sin límite práctico de duración (probado con 2+ horas)
- **4 formatos de salida**: texto con timestamps, texto limpio, SRT (subtítulos), JSON
- **Interfaz web** moderna con drag & drop, progreso en tiempo real
- **API REST** para integración con otros servicios
- **Docker ready** con soporte GPU/CPU

---

## Inicio Rápido

### Opción 1: Local (Python)

```bash
# 1. Clonar e instalar
git clone <tu-repo>/transcribeya.git
cd transcribeya
pip install -r requirements.txt

# 2. Asegurar que tienes ffmpeg
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: choco install ffmpeg

# 3. Ejecutar
python app.py

# Abre http://localhost:8000
```

### Opción 2: Docker

```bash
# CPU (modelo medium)
docker compose up -d

# GPU (modelo large, mejor calidad)
docker compose --profile gpu up -d
```

### Variables de Entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `WHISPER_MODEL` | `small` | Modelo: `tiny`, `base`, `small`, `medium`, `large` |
| `PORT` | `8000` | Puerto del servidor |
| `MAX_FILE_SIZE_MB` | `500` | Tamaño máximo de archivo |
| `FREE_MINUTES_LIMIT` | `30` | Minutos gratis (para implementar cobro) |

### Modelos Disponibles

| Modelo | VRAM | Velocidad* | Calidad | Recomendado para |
|--------|------|-----------|---------|------------------|
| `tiny` | ~1 GB | 32x | ★★☆☆☆ | Pruebas rápidas |
| `base` | ~1 GB | 16x | ★★★☆☆ | Audio claro, pocos hablantes |
| `small` | ~2 GB | 6x | ★★★½☆ | Uso general |
| `medium` | ~5 GB | 2x | ★★★★☆ | **Mejor balance calidad/velocidad** |
| `large` | ~10 GB | 1x | ★★★★★ | Máxima precisión, requiere GPU |

*Velocidad relativa al tiempo real en GPU. En CPU es ~4-10x más lento.

---

## API

### POST `/api/upload`

Sube un archivo y comienza la transcripción.

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@grabacion.mp3" \
  -F "language=es"
```

Respuesta:
```json
{
  "job_id": "a1b2c3d4",
  "duration_minutes": 45.2,
  "file_size_mb": 128.3,
  "message": "Transcripción iniciada"
}
```

### GET `/api/status/{job_id}`

Consulta el estado de la transcripción.

### GET `/api/download/{job_id}/{format}`

Descarga el resultado. Formatos: `txt`, `clean_txt`, `srt`, `json`.

### GET `/api/health`

Estado del servidor (modelo cargado, GPU disponible, jobs activos).

---

## Despliegue en Producción

### Railway / Render (más fácil, CPU)

1. Sube el repo a GitHub
2. Conecta en [Railway](https://railway.app) o [Render](https://render.com)
3. Variables: `WHISPER_MODEL=small`, `PORT=8000`
4. Deploy automático

> **Nota**: En CPU el modelo `small` es la mejor opción calidad/costo.
> Un audio de 1 hora tarda ~15-25 min en transcribirse con `small` en CPU.

### RunPod / Vast.ai (GPU, mejor rendimiento)

1. Usa el Dockerfile
2. Selecciona instancia con GPU (RTX 3060+ recomendado)
3. `WHISPER_MODEL=large` para máxima calidad
4. Un audio de 1 hora tarda ~3-5 min con `large` en GPU

### VPS propio (DigitalOcean, Hetzner, etc.)

```bash
# En servidor con 8GB+ RAM
sudo apt install docker.io docker-compose
git clone <tu-repo>
cd transcribeya
docker compose up -d
```

---

## Modelo de Negocio

### Costos Estimados de Infraestructura

| Plataforma | Tipo | Costo/mes | Capacidad estimada |
|------------|------|-----------|-------------------|
| Railway (free) | CPU | $0 | ~5 hrs audio/mes |
| Railway (pro) | CPU | ~$5-20 | ~20-50 hrs audio/mes |
| Render | CPU | ~$7-25 | ~20-50 hrs audio/mes |
| Hetzner VPS | CPU 8GB | ~$7 | ~30-60 hrs audio/mes |
| RunPod spot | GPU | ~$0.20/hr | Bajo demanda |
| Vast.ai | GPU | ~$0.10-0.30/hr | Bajo demanda |

### Opciones de Monetización

#### A) Freemium (recomendado para empezar)

- **Gratis**: Hasta 30 min por archivo, modelo `small`
- **Pro** ($3-5 USD/mes o pago único por archivo):
  - Sin límite de duración
  - Modelo `large` (mejor calidad)
  - Prioridad en cola
  - Detección de hablantes (futuro)

#### B) Pago por uso

- $0.01-0.05 USD por minuto de audio
- Comparable con servicios como Otter.ai ($0.01/min) o Rev ($0.25/min humano)
- Integrar con Stripe/MercadoPago

#### C) Donación / Pay What You Want

- Botón de [Buy Me a Coffee](https://buymeacoffee.com)
- Ko-fi, PayPal.me, o MercadoPago link
- Funciona bien si el tráfico es bajo-medio

#### D) Híbrido (lo más práctico)

Gratis con donación voluntaria + tier Pro para uso intensivo.

### Implementar Pagos (ejemplo con Stripe)

Para integrar pagos, necesitarías agregar:

1. **Verificar `is_free`** en el response de `/api/upload`
2. Si `is_free == false`, redirigir a Stripe Checkout antes de iniciar transcripción
3. Guardar receipt y liberar el job

El campo `FREE_MINUTES_LIMIT` ya está implementado en el código.

---

## Roadmap Futuro

- [ ] Detección automática de hablantes (diarización con pyannote)
- [ ] Cola de trabajos con Celery/Redis (para múltiples usuarios)
- [ ] Autenticación de usuarios
- [ ] Panel de administración con métricas
- [ ] Integración con MercadoPago/Stripe
- [ ] Traducción automática post-transcripción
- [ ] Resumen automático con LLM

---

## Licencia

MIT — Usa, modifica y distribuye libremente.
Whisper es de OpenAI bajo licencia MIT.
