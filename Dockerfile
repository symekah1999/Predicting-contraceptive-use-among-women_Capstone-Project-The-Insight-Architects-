# ── KDHS 2022 Contraceptive Use API ───────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by LightGBM's OpenMP dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py feature_engineering.py contraceptive_model_bundle.joblib ./
COPY templates ./templates

# API key — override at runtime with -e API_KEY=...
# Leaving it empty here is intentional: the app generates a random key
# at startup if none is provided (and prints it to the logs).
ENV API_KEY=""
ENV PORT=5000

EXPOSE 5000

# Run with gunicorn (production WSGI server — NOT Flask dev server)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "60", "app:app"]
