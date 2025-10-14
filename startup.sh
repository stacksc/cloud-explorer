#!/usr/bin/env bash
set -e

echo "🚀 Custom startup script running..."
cd /home/site/wwwroot

echo "📁 Current directory: $(pwd)"
echo "🐍 System Python: $(which python)"

# Ensure venv exists
if [ ! -d "antenv" ]; then
  echo "🔧 Creating virtual environment..."
  python3 -m venv antenv
fi

# Activate venv
source antenv/bin/activate
echo "🐍 After activating venv: $(which python)"

# 🧹 Clean up Azure’s /agents/python interference
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PYTHONPATH//\/agents\/python:/}"
fi
export PYTHONPATH="/home/site/wwwroot/antenv/lib/python3.11/site-packages:$PYTHONPATH"

# 🧩 Ensure core packages are present (idna, httpx, typing-extensions)
echo "📦 Ensuring dependencies..."
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir --upgrade -r requirements.txt
pip install --no-cache-dir --upgrade idna httpx typing-extensions

python -c "import sys, typing_extensions, idna, httpx; \
print('🔍 sys.path[:5]=', sys.path[:5]); \
print('✅ typing_extensions from:', typing_extensions.__file__); \
print('✅ idna version:', idna.__version__); \
print('✅ httpx version:', httpx.__version__)"

# Runtime environment
export FLASK_RUN_PORT=${PORT:-8000}
export GUNICORN_CMD_ARGS="--timeout 600 --workers 1 --worker-class gthread --threads 4 --log-level info"

echo "✅ Launching Gunicorn..."
exec /home/site/wwwroot/antenv/bin/gunicorn -b 0.0.0.0:$FLASK_RUN_PORT flask_server:app

