# --- Builder Stage ---
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build tools (only needed for compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Install CPU-only versions of torch to save GBs of space
# 2. Install dependencies to a specific folder for easy copying
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download the model to a specific, known path
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/model_cache')"

# --- Final Stage ---
FROM python:3.10-slim

WORKDIR /app

# Copy ONLY the installed site-packages and the specific model cache
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/model_cache /app/model_cache

# Copy application code
COPY run.py api.py ./

# Set Environment Variables
ENV PATH=/root/.local/bin:$PATH
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
# Ensure python uses the CPU-only libs
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]