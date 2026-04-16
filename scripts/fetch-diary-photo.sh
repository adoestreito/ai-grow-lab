#!/usr/bin/env bash
# Downloads one JPEG from the ESP32 camera web server into ~/Diary/Photos.
# Set ESP32_CAM_URL to your board's HTTP base (no trailing slash), e.g. http://192.168.1.50

set -euo pipefail

ESP32_CAM_URL="${ESP32_CAM_URL:-http://192.168.1.100}"
DEST_DIR="${DEST_DIR:-$HOME/Diary/Photos}"

mkdir -p "$DEST_DIR"
stamp="$(date +%Y-%m-%d_%H-%M-%S)"
out="$DEST_DIR/diary_${stamp}.jpg"

if ! curl -fsS --connect-timeout 15 --max-time 120 \
  "${ESP32_CAM_URL}/capture" -o "$out"; then
  echo "fetch-diary-photo: failed to download capture" >&2
  exit 1
fi

echo "Saved $out"
