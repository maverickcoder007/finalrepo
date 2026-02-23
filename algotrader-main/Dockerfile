FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./

RUN uv pip install --system --no-cache .

COPY . .

RUN mkdir -p logs data/historical

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
