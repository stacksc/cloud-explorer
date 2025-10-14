#!/usr/bin/env bash
set -euo pipefail

APP_NAME="cloud-explorer"
RESOURCE_GROUP="myRG"
ZIP_NAME="deploy_package.zip"
VENV_DIR="antenv"

echo "🚀 Starting deployment to Azure WebApp: $APP_NAME"

# 1. Handle virtual environment
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "⚠️  Detected active venv at: $VIRTUAL_ENV (ignoring outer venv)"
  unset VIRTUAL_ENV
fi

# 2. Create or reuse venv
if [[ ! -d "$VENV_DIR" ]]; then
  echo "🐍 Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
else
  echo "✅ Using existing virtual environment: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# 3. Dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# 4. Clean up and package
echo "🧹 Cleaning up..."
rm -f "$ZIP_NAME"
find . -type d -name "__pycache__" -exec rm -rf {} +
echo "📦 Creating zip package..."
zip -r "$ZIP_NAME" . -x \
  "*.git*" "*.DS_Store" "*__pycache__*" "*.pytest_cache*" \
  "*.venv*" "*antenv*" "*.idea*" "*uploads*" "*.log"

# 5. Delay to avoid SCM restarts
echo "⏳ Waiting 45 seconds to ensure SCM is stable..."
sleep 45

# 6. Deploy using new command (replaces config-zip)
echo "🚀 Deploying $ZIP_NAME to $APP_NAME ..."
if ! az webapp deploy \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --src-path "$ZIP_NAME" \
    --type zip; then
  echo "⚠️ Deployment failed on first attempt. Retrying after short delay..."
  sleep 30
  az webapp deploy \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --src-path "$ZIP_NAME" \
    --type zip
fi

echo "✅ Deployment complete."
