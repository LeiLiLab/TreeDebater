"""
Given a chunk_profile.csv, visualize the streaming overlap timeline.

Streaming model assumed:
  - chunk 0: TTS call first, then starts playing
  - chunk i prep (refine + TTS) runs during chunk i-1 playback
  - if prep finishes before chunk i-1 ends → seamless
  - if prep overruns → gap (silence) inserted

Usage:
  python overlap_viz.py outputs_20260320_152412/closing_against/chunk_profile.csv
  python overlap_viz.py outputs_20260320_152412/closing_against/chunk_profile.csv -o my_fig.png
"""

import ast
import csv
import sys
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def load_profile(csv_path):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                if v in ("True", "False"):
                    parsed[k] = v == "True"
                elif v.startswith("["):
                    parsed[k] = ast.literal_eval(v)
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def compute_timeline(rows):
    events = []

    # chunk 0: refine + TTS both happen before audio starts (nothing plays in parallel)
    t = 0.0
    refine0_start = t
    refine0_end   = t + rows[0]["refine_total_s"]
    tts0_start    = refine0_end
    tts0_end      = tts0_start + rows[0]["tts_api_s"]
    play0_start   = tts0_end
    play0_end     = play0_start + rows[0]["audio_seconds"]

    events.append({
        "chunk": 0,
        "refine_start": refine0_start,
        "refine_end":   refine0_end,
        "refine_segments": _build_segments(rows[0], refine0_start),
        "tts_start":    tts0_start,
        "tts_end":      tts0_end,
        "play_start":   play0_start,
        "play_end":     play0_end,
        "gap": 0.0,
        "overrun": 0.0,
    })

    for i in range(1, len(rows)):
        # prep starts when previous chunk starts playing
        prep_start = events[i - 1]["play_start"]
        refine_s = rows[i]["refine_total_s"]
        tts_s = rows[i]["tts_api_s"]

        refine_start = prep_start
        refine_end = prep_start + refine_s
        tts_start = refine_end
        tts_end = tts_start + tts_s

        prev_play_end = events[i - 1]["play_end"]
        ready_at = tts_end
        overrun = max(0.0, ready_at - prev_play_end)   # prep took longer than prev playback
        gap = max(0.0, ready_at - prev_play_end)

        play_start = max(prev_play_end, ready_at)
        play_end = play_start + rows[i]["audio_seconds"]

        events.append({
            "chunk": i,
            "refine_start": refine_start,
            "refine_end": refine_end,
            "refine_segments": _build_segments(rows[i], refine_start),
            "tts_start": tts_start,
            "tts_end": tts_end,
            "play_start": play_start,
            "play_end": play_end,
            "gap": gap,
            "overrun": overrun,
        })

    return events


def _build_segments(row, refine_start):
    """
    Returns a list of (start, end, kind) tuples for the refinement sub-steps.
    Pattern: fs[0], llm[0], fs[1], llm[1], ...
    kind is 'fs' or 'llm'.
    """
    fs_times  = row.get("iter_fs_times_s", []) or []
    llm_times = row.get("iter_llm_times_s", []) or []
    segments = []
    t = refine_start
    n_iters = max(len(fs_times), len(llm_times))
    for j in range(n_iters):
        if j < len(fs_times):
            segments.append((t, t + fs_times[j], "fs"))
            t += fs_times[j]
        if j < len(llm_times):
            segments.append((t, t + llm_times[j], "llm"))
            t += llm_times[j]
    return segments


def plot_timeline(rows, events, out_path, title="Chunk Overlap Timeline"):
    n = len(rows)

    COLORS = {
        "refine": "#DD8452",
        "fs":     "#DD8452",   # orange – FS estimation
        "llm":    "#8172B2",   # purple – LLM refinement
        "tts":    "#4C72B0",
        "play":   "#55A868",
        "gap":    "#C44E52",
    }

    BAR_H = 0.38
    Y_PLAY = 1.0
    Y_PREP = 0.0

    fig, ax = plt.subplots(figsize=(max(14, n * 2.5), 4.5))

    for ev in events:
        i = ev["chunk"]

        # ── playback bar ──────────────────────────────────────────────────────
        dur = ev["play_end"] - ev["play_start"]
        ax.barh(Y_PLAY, dur, left=ev["play_start"], height=BAR_H,
                color=COLORS["play"], alpha=0.88, edgecolor="white", linewidth=0.6)
        ax.text(ev["play_start"] + dur / 2, Y_PLAY,
                f"▶ {i}  {dur:.1f}s",
                ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")

        # ── gap bar (silence) ────────────────────────────────────────────────
        if ev["gap"] > 0.05:
            gap_left = ev["play_start"] - ev["gap"]
            ax.barh(Y_PLAY, ev["gap"], left=gap_left, height=BAR_H,
                    color=COLORS["gap"], alpha=0.8, edgecolor="white", linewidth=0.6)
            ax.text(gap_left + ev["gap"] / 2, Y_PLAY,
                    f"⚠ {ev['gap']:.1f}s",
                    ha="center", va="center", fontsize=7.5, color="white")

        # ── refinement bar (sub-segments: fs / llm alternating) ─────────────
        segments = ev.get("refine_segments", [])
        if segments:
            for seg_start, seg_end, kind in segments:
                seg_dur = seg_end - seg_start
                if seg_dur < 0.05:
                    continue
                ax.barh(Y_PREP, seg_dur, left=seg_start, height=BAR_H,
                        color=COLORS[kind], alpha=0.88, edgecolor="white", linewidth=0.6)
                ax.text(seg_start + seg_dur / 2, Y_PREP,
                        f"{kind}\n{seg_dur:.1f}s",
                        ha="center", va="center", fontsize=7, color="white")
        elif ev["refine_start"] is not None:
            ref_dur = ev["refine_end"] - ev["refine_start"]
            if ref_dur > 0.05:
                ax.barh(Y_PREP, ref_dur, left=ev["refine_start"], height=BAR_H,
                        color=COLORS["refine"], alpha=0.88, edgecolor="white", linewidth=0.6)
                n_ref = int(rows[i].get("n_ref_used", 0))
                ax.text(ev["refine_start"] + ref_dur / 2, Y_PREP,
                        f"refine {i}({n_ref})\n{ref_dur:.1f}s",
                        ha="center", va="center", fontsize=7.5)

        # ── TTS API bar ──────────────────────────────────────────────────────
        tts_dur = ev["tts_end"] - ev["tts_start"]
        ax.barh(Y_PREP, tts_dur, left=ev["tts_start"], height=BAR_H,
                color=COLORS["tts"], alpha=0.88, edgecolor="white", linewidth=0.6)
        ax.text(ev["tts_start"] + tts_dur / 2, Y_PREP,
                f"TTS {i}\n{tts_dur:.1f}s",
                ha="center", va="center", fontsize=7.5, color="white")

    # chunk boundary lines (at each play_start)
    for ev in events:
        ax.axvline(ev["play_start"], color="gray", linewidth=0.9,
                   linestyle="--", alpha=0.45, zorder=0)

    # end-of-playback line
    total_end = events[-1]["play_end"]
    ax.axvline(total_end, color="black", linewidth=1.8, linestyle="-", zorder=3)
    ax.text(total_end, 1.45, f" ⏹ {total_end:.1f}s",
            ha="left", va="top", fontsize=9, fontweight="bold", color="black")

    ax.set_yticks([Y_PREP, Y_PLAY])
    ax.set_yticklabels(["Prep\n(refine + TTS API)", "Playback"], fontsize=10)
    ax.set_xlabel("Wall-clock time (s)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(-0.45, 1.55)
    ax.grid(axis="x", alpha=0.25)

    legend_patches = [
        mpatches.Patch(color=COLORS["play"],   label="Audio playback"),
        mpatches.Patch(color=COLORS["fs"],     label="FS estimation"),
        mpatches.Patch(color=COLORS["llm"],    label="LLM refinement"),
        mpatches.Patch(color=COLORS["tts"],    label="TTS API call"),
        mpatches.Patch(color=COLORS["gap"],    label="Gap / silence"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


def print_summary(rows, events):
    print(f"\n{'chunk':>6}  {'play_start':>10}  {'play_end':>8}  "
          f"{'gap_s':>6}  {'overrun_s':>9}  {'refine_s':>9}  {'tts_s':>6}  {'audio_s':>7}")
    print("-" * 78)
    for ev in events:
        i = ev["chunk"]
        print(f"{i:>6}  {ev['play_start']:>10.2f}  {ev['play_end']:>8.2f}  "
              f"{ev['gap']:>6.2f}  {ev['overrun']:>9.2f}  "
              f"{rows[i]['refine_total_s']:>9.2f}  "
              f"{rows[i]['tts_api_s']:>6.2f}  "
              f"{rows[i]['audio_seconds']:>7.3f}")
    total_play = events[-1]["play_end"] - events[0]["play_start"]
    total_audio = sum(r["audio_seconds"] for r in rows)
    total_gap = sum(ev["gap"] for ev in events)
    print("-" * 78)
    print(f"  Total playback span : {total_play:.2f}s")
    print(f"  Total audio content : {total_audio:.2f}s")
    print(f"  Total gap / silence : {total_gap:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to chunk_profile.csv")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_path = Path(args.output) if args.output else csv_path.parent / "overlap_timeline.png"

    rows = load_profile(csv_path)
    events = compute_timeline(rows)

    title = f"Chunk Overlap Timeline — {csv_path.parent.name}"
    plot_timeline(rows, events, out_path, title=title)
    print_summary(rows, events)
