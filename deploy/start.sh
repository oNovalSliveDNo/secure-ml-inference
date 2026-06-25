#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-80}"
APP_MODE="${APP_MODE:-api}"

case "${APP_MODE}" in
  api)
    echo "Starting FastAPI on port ${PORT}"
    exec uvicorn api.main:app \
      --host 0.0.0.0 \
      --port "${PORT}"
    ;;

  ui)
    echo "Starting Streamlit on port ${PORT}"
    exec streamlit run ui/streamlit_app.py \
      --server.address 0.0.0.0 \
      --server.port "${PORT}" \
      --server.headless true \
      --browser.gatherUsageStats false
    ;;

  *)
    echo "Unknown APP_MODE=${APP_MODE}. Expected 'api' or 'ui'."
    exit 1
    ;;
esac