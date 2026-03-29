FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* README.md ./

RUN pip install --upgrade pip setuptools wheel && \
    pip install uv && \
    uv pip install --system .

COPY backend ./backend
COPY src ./src
COPY logger ./logger
COPY feature_store ./feature_store
COPY main.py ./
COPY system_design.md ./

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
