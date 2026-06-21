#!/usr/bin/env bash
set -euo pipefail

# Requires Kaggle API credentials:
#   1. Go to https://www.kaggle.com/settings
#   2. Create an API token
#   3. Save it as ~/.kaggle/kaggle.json
#   4. chmod 600 ~/.kaggle/kaggle.json
#
# Run:
#   bash backend/scripts/download_public_datasets.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_DIR="$ROOT_DIR/data/raw"

mkdir -p "$RAW_DIR"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install it with: python -m pip install kaggle"
  exit 1
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "Missing ~/.kaggle/kaggle.json. Create a Kaggle API token first."
  exit 1
fi

download_dataset() {
  local slug="$1"
  local out="$2"
  mkdir -p "$RAW_DIR/$out"
  echo "Downloading $slug -> data/raw/$out"
  kaggle datasets download -d "$slug" -p "$RAW_DIR/$out" --unzip
}

# Strong starter datasets for the current project.
# These are useful, but you should inspect licensing/terms in Kaggle before final submission.
download_dataset "andrewmvd/helmet-detection" "helmet_detection"
download_dataset "andrewmvd/car-plate-detection" "car_plate_detection"
download_dataset "dataclusterlabs/indian-number-plates-dataset" "indian_number_plates"

cat <<'MSG'

Downloaded starter datasets.

Next:
  python backend/scripts/convert_voc_to_yolo.py \
    --input data/raw/helmet_detection \
    --output data/processed/helmet_yolo \
    --class-map helmet=helmet head=no_helmet person=person motorcycle=motorcycle

Inspect the raw annotation class names first; every dataset uses different labels.
MSG
