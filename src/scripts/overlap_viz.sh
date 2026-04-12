#!/bin/bash
# Usage: bash overlap_viz.sh [chunks_dir_pattern]
# Default: all _*_chunks/ directories under src/

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATTERN="${1:-$SCRIPT_DIR/_*_chunks}"

found=0
for dir in $PATTERN; do
    csv="$dir/chunk_profile.csv"
    if [ -f "$csv" ]; then
        echo "[VIZ] $csv"
        python "$SCRIPT_DIR/overlap_viz_par.py" "$csv"
        found=$((found + 1))
    else
        echo "[SKIP] $csv not found"
    fi
done

if [ "$found" -eq 0 ]; then
    echo "No chunk_profile.csv found. Run with streaming_tts: true first."
fi
