# Single-origin image: FastAPI API + static frontend (used for the AWS deployment).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SERVE_FRONTEND=true \
    FRONTEND_DIR=/app/frontend

WORKDIR /app
COPY backend/requirements.txt .
RUN python -m pip install --retries 10 --prefer-binary -r requirements.txt

COPY backend/ /app/
COPY frontend/ /app/frontend/

RUN adduser --disabled-password --no-create-home appuser \
 && mkdir -p /app/data/uploads \
 && chown -R appuser:appuser /app/data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT', '8000'))" || exit 1
# Never run a public instance on the built-in default JWT secret: generate one per boot
# unless JWT_SECRET is supplied (e.g. as a hosting-platform secret).
CMD ["sh", "-c", "export JWT_SECRET=\"${JWT_SECRET:-$(python -c 'import secrets; print(secrets.token_hex(32))')}\"; exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]
