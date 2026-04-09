# ── Stage 1: base ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps (needed by some tokenizer wheels)
RUN apt-get update && apt-get install -y 
WORKDIR /app

# ── Stage 2: dependencies ───────────────────────────────────────────────────────
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the model so the container starts instantly at runtime.
# Change MODEL_NAME build-arg to use a different model.
ARG MODEL_NAME=all-MiniLM-L6-v2
ENV MODEL_NAME=${MODEL_NAME}

RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer
import os
SentenceTransformer(os.environ["MODEL_NAME"])
print("Model cached successfully.")
EOF

# ── Stage 3: application ────────────────────────────────────────────────────────
COPY run.py  .
COPY api.py  .

# Uvicorn will listen on this port (overridable via ENV)
ENV PORT=8000
EXPOSE 8000

# Entrypoint
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 2"]
