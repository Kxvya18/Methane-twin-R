FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgdal-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY api ./api
COPY simulator ./simulator
COPY configs ./configs

RUN pip install --no-cache-dir -e ".[research]"

CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
