#!/bin/bash
# Usage: bash overlap_viz.sh [chunks_dir_pattern]
# Default: all _*_chunks/ directories under src/
# When 2+ chunks dirs are matched, also produce a combined top-to-bottom PNG
# (overlap_timeline_combined.png) in the common parent directory.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATTERN="${@:-$SCRIPT_DIR/_*_chunks}"

pngs=()
found=0
for dir in $PATTERN; do
    csv="$dir/chunk_profile.csv"
    if [ -f "$csv" ]; then
        echo "[VIZ] $csv"
        python "$SCRIPT_DIR/overlap_viz_par.py" "$csv"
        png="$dir/overlap_timeline_par.png"
        if [ -f "$png" ]; then
            pngs+=("$png")
        fi
        found=$((found + 1))
    else
        echo "[SKIP] $csv not found"
    fi
done

if [ "$found" -eq 0 ]; then
    echo "No chunk_profile.csv found. Run with streaming_tts: true first."
elif [ "${#pngs[@]}" -ge 2 ]; then
    parent=$(python -c "
import os, sys
print(os.path.commonpath([os.path.dirname(p) for p in sys.argv[1:]]))
" "${pngs[@]}")
    out="$parent/overlap_timeline_combined.png"
    python "$SCRIPT_DIR/stack_pngs.py" "${pngs[@]}" -o "$out"
fi
