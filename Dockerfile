FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEEP_RESEARCH_CACHE=/data/cache \
    DEEP_RESEARCH_STATS=/data/stats.db \
    PORT=8000

WORKDIR /app

# Install deps first (cached layer)
COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy source
COPY *.py ./

RUN useradd --create-home --uid 1000 app && \
    mkdir -p /data/cache && chown -R app:app /data
USER app

VOLUME ["/data"]
EXPOSE 8000

# Healthcheck hits /healthz
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz', timeout=2).status==200 else 1)" || exit 1

CMD ["uvicorn", "web:app", "--host", "0.0.0.0", "--port", "8000"]
