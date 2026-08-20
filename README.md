# RevisAI

AI weak-topic diagnostic quiz generator. Paste or photograph your notes for 3–5 topics,
answer a generated quiz, and see a confidence-vs-accuracy breakdown per topic.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional but recommended: OCR for uploaded photos of notes
# macOS:   brew install tesseract
# Ubuntu:  sudo apt install tesseract-ocr
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

export SECRET_KEY="something-random"       # Windows: set SECRET_KEY=...
export GEMINI_API_KEY="your-key-here"      # optional — pasted-note MCQs work without it
python app.py
```

Visit http://localhost:7860. Without Tesseract installed, the photo-upload path returns
no text — use the "paste text" field for each topic instead.

## Deploy

The app is a standard Flask app served by gunicorn. Any host that runs a Docker
container or a Python web service works.

### Option A — Docker (recommended, includes OCR)

The Dockerfile installs the `tesseract-ocr` system package, which `pytesseract`
needs — without it, photo uploads won't extract any text.

```bash
docker build -t revisai .
docker run -p 7860:7860 -e SECRET_KEY=... -e GEMINI_API_KEY=... revisai
```

Push this image to any container host: Render (New → Web Service → Docker),
Railway, Fly.io, Google Cloud Run, or a VPS.

### Option B — Native Python host (Procfile), no OCR

Platforms like Render or Railway can also run the `Procfile` directly without
Docker. This is simpler but usually can't install `tesseract-ocr`, so leave
photo upload aside and rely on the "paste text" field.

1. Push this folder to a GitHub repo.
2. On Render/Railway: New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: (from `Procfile`, usually auto-detected) `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Set environment variables `SECRET_KEY` and (optionally) `GEMINI_API_KEY`.

### Environment variables

| Variable         | Required | Purpose                                                        |
|------------------|----------|------------------------------------------------------------------|
| `SECRET_KEY`     | Yes (prod) | Flask session signing key — set a random string                |
| `GEMINI_API_KEY` | No       | Enables AI-written questions via Gemini; falls back to the offline engine if unset |
| `PORT`           | No       | Defaults to 7860; most hosts set this automatically             |

## Known limitations

- **Sessions live in memory** (`SESSIONS` dict in `app.py`), not a database. That's fine for a
  single-process demo, but it means: (a) restarting the app clears everyone's quizzes/history,
  and (b) it will not work correctly if the host scales you to more than one process/worker —
  a user's requests could land on a worker that never saw their data. Keep worker count at 1
  (already set in the Dockerfile/Procfile) unless you swap in a real session store
  (e.g. Redis via `Flask-Session`).
- **OCR requires the Tesseract binary**, not just the Python package. Use the Docker image, or
  install `tesseract-ocr` on the host yourself, or skip photo upload and paste notes as text.
- The offline "semantic question synthesizer" recognizes a handful of subject domains
  (physics/mechanics, Python, cell biology, OS/deadlocks, plus a generic fallback) — topics
  outside those get generic cloze-style questions unless a Gemini API key is set.
