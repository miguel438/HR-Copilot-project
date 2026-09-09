# Single image reused by three services (agent / ui / ingest) via docker-compose command
# overrides, so there is one dependency set and one build to keep in sync.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 5000 8501

# Default command runs the Flask API (the `agent` service). `--chdir src` puts src/ on the
# import path so app.py's sibling imports (graph, config, ...) resolve the same way they do
# when run locally with `python app.py` from inside src/. The `ui` and `ingest` services in
# docker-compose.yml override this CMD with their own entrypoint.
# --timeout 300 matches the timeout on n8n's HTTP Request node, so the two ends of the same call
# give up at the same point rather than one reporting a failure the other is still working on.
CMD ["gunicorn", "--chdir", "src", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "app:app"]
