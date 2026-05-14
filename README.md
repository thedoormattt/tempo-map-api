# tempo-map-api

FastAPI backend for Tempo Map. Uses librosa for high-accuracy BPM and key detection.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/analyse` | Analyse an audio file |

### POST /analyse

**Request:** `multipart/form-data` with a single `file` field containing an audio file (MP3, WAV, FLAC, OGG, M4A, AAC).

**Response:**
```json
{
  "bpm_curve":   [{ "t": 2.0, "bpm": 124.5 }, ...],
  "key_curve":   [{ "t": 4.0, "key": "A minor", "root": 9, "mode": "minor", "confidence": 0.87 }, ...],
  "overall_key": { "label": "A minor", "root": 9, "mode": "minor", "confidence": 0.87 },
  "duration":    214.3,
  "sample_rate": 44100,
  "channels":    1
}
```

## Local development

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

API will be at http://localhost:8000. Swagger docs at http://localhost:8000/docs.

## Deploy to Render

1. Push this directory to a GitHub repo (can be the same monorepo as the frontend, or separate)
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect the repo and point the root to this directory
4. Render will detect `render.yaml` automatically
5. In the Render dashboard, set the `ALLOWED_ORIGINS` environment variable to your Vercel frontend URL, e.g. `https://tempo-map.vercel.app`

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ALLOWED_ORIGINS` | Comma-separated list of allowed CORS origins | `*` (allow all — restrict in production) |
| `PORT` | Set automatically by Render | `8000` |
