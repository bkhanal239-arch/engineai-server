#!/bin/bash
set -e

echo "=== EngineAI — running ingestion ==="
python ingest.py

echo "=== Starting server on port ${PORT:-8000} ==="
exec python server.py
