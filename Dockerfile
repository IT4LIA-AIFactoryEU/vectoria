FROM python:3.12-slim

# System dependencies:
#   libgl1 + libglib2.0-0  → OpenCV (cv2) used by Docling table model
#   tesseract-ocr           → OCR engine for scanned PDFs (Docling)
#   tesseract-ocr-ita       → Italian language pack for Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-ita \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd -m -u 1000 -s /bin/bash vectoria

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install all runtime deps + api extra (fastapi + uvicorn), skip dev/test/ui
RUN uv sync --extra api --extra observability --extra docling --extra graph --no-dev --frozen

# Copy the rest of the project
COPY . .

# Re-install the project itself in editable mode
RUN uv sync --extra api --extra observability --extra docling --extra graph --no-dev --frozen

# spaCy model for the graph relation-label fallback (graph_store.relation_extraction: false).
# The 'spacy' package alone ships no language models — must be fetched separately.
RUN uv run python -m spacy download it_core_news_sm

# Create data directories and hand ownership to the non-root user.
# The actual volume mounts happen at runtime; pre-creating the dirs ensures
# the correct owner when Docker initialises an empty named volume.
RUN mkdir -p /data/index /data/uploads && \
    chown -R vectoria:vectoria /data /app

USER vectoria

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command — override INDEX_PATH and CONFIG_PATH via environment or compose
CMD uv run python bin/vectoria \
    --config ${CONFIG_PATH:-etc/docker/config.yaml} \
    serve \
    ${INDEX_PATH:+--index-path ${INDEX_PATH}} \
    --index-dir /data/index \
    --upload-dir /data/uploads \
    --host 0.0.0.0 \
    --port 8000
