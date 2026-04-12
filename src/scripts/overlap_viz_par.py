"""
Parallel-aware chunk overlap timeline visualizer CSV output.

Three lanes (y-axis):
  Main thread  — fs / llm sub-segments (orange / purple)
  TTS band     — one sub-row per submitted candidate; chosen = dark blue,
                 background = light blue, still-running = grey dashed arrow
  Playback     — audio duration (green) + gap/silence (red)

Timeline rules (same streaming model as overlap_viz.py):
  chunk 0  : prep = TTS only (no refinement); play starts when TTS finishes
  chunk i≥1: prep starts when chunk i-1 starts playing;
             prep wall-clock = total_elapsed_s (covers refine + any TTS wait);
             gap inserted if prep overruns chunk i-1 playback

TTS candidate submit times are derived from iter_fs_times_s / iter_llm_times_s:
  candidate k submitted after  sum(fs[0..k]) + sum(llm[0..k-1])

Usage:
  python overlap_viz_par.py outputs_20260403_120000/closing_for/chunk_profile.csv
  python overlap_viz_par.py ... -o my_fig.png
"""

import ast
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ── data loading ──────────────────────────────────────────────────────────────

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


# ── timeline computation ──────────────────────────────────────────────────────

def compute_timeline(rows):
    events = []

    # chunk 0: TTS only, no refinement
    t = 0.0
    play0_start = t + rows[0]["total_elapsed_s"]   # = tts_api_s for chunk 0
    play0_end   = play0_start + rows[0]["audio_seconds"]

    events.append({
        "chunk":          0,
        "prep_start":     t,
        "main_segments":  [],                             # no fs/llm
        "tts_candidates": _build_tts_candidates(rows[0], t),
        "play_start":     play0_start,
        "play_end":       play0_end,
        "gap":            0.0,
    })

    for i in range(1, len(rows)):
        prep_start    = events[i - 1]["play_start"]
        prep_duration = rows[i]["total_elapsed_s"]        # true wall-clock (parallel)

        prev_play_end = events[i - 1]["play_end"]
        ready_at      = prep_start + prep_duration
        gap           = max(0.0, ready_at - prev_play_end)
        play_start    = max(prev_play_end, ready_at)
        play_end      = play_start + rows[i]["audio_seconds"]

        events.append({
            "chunk":          i,
            "prep_start":     prep_start,
            "main_segments":  _build_main_segments(rows[i], prep_start),
            "tts_candidates": _build_tts_candidates(rows[i], prep_start),
            "play_start":     play_start,
            "play_end":       play_end,
            "gap":            gap,
        })

    return events


def _build_main_segments(row, prep_start):
    """fs / llm alternating segments on the main thread."""
    fs_times  = row.get("iter_fs_times_s",  []) or []
    llm_times = row.get("iter_llm_times_s", []) or []
    segments = []
    t = prep_start
    n_iters = max(len(fs_times), len(llm_times))
    for j in range(n_iters):
        if j < len(fs_times):
            segments.append((t, t + fs_times[j], "fs"))
            t += fs_times[j]
        if j < len(llm_times):
            segments.append((t, t + llm_times[j], "llm"))
            t += llm_times[j]
    return segments


def _build_tts_candidates(row, prep_start):
    """
    Return list of dicts {k, submit_t, duration, is_chosen}.

    Candidate k is submitted after  sum(fs[0..k]) + sum(llm[0..k-1])
    duration == -1.0 means the future had not completed at selection time.
    """
    fs_times  = row.get("iter_fs_times_s",  []) or []
    llm_times = row.get("iter_llm_times_s", []) or []
    tts_times = row.get("iter_tts_times_s", []) or []
    used_iter = int(row.get("used_candidate_iter", 0))

    result = []
    for k, tts_dur in enumerate(tts_times):
        submit_t = prep_start + sum(fs_times[: k + 1]) + sum(llm_times[:k])
        result.append({
            "k":         k,
            "submit_t":  submit_t,
            "duration":  float(tts_dur),
            "is_chosen": k == used_iter,
        })
    return result


# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "fs":     "#DD8452",   # orange  – FastSpeech estimation
    "llm":    "#8172B2",   # purple  – LLM refinement
    "play":   "#55A868",   # green   – audio playback
    "gap":    "#C44E52",   # red     – gap / silence
    "tts_ok": "#4C72B0",   # dark blue  – chosen TTS candidate
    "tts_bg": "#AEC6E8",   # light blue – background TTS candidate (done)
    "tts_nd": "#BBBBBB",   # grey       – TTS still running at selection
}

BAR_H   = 0.32   # playback / main-thread bar height
TTS_H   = 0.18   # height of each TTS candidate sub-row
TTS_GAP = 0.06   # vertical gap between candidate sub-rows
Y_MAIN  = 0.0    # centre of main-thread lane


def _tts_y(k):
    """Centre y for TTS candidate k."""
    return 1.1 + k * (TTS_H + TTS_GAP)


def _y_play(max_cands):
    """Centre y for playback lane (above TTS band)."""
    top_of_tts = _tts_y(max_cands - 1) + TTS_H / 2
    return top_of_tts + 0.55


def plot_timeline(rows, events, out_path, title="Parallel Chunk Overlap Timeline"):
    n = len(rows)
    max_cands = max((len(ev["tts_candidates"]) for ev in events), default=1)
    Y_PLAY    = _y_play(max_cands)

    fig_h = max(5.0, Y_PLAY + 1.2)
    fig, ax = plt.subplots(figsize=(max(14, n * 2.5), fig_h))

    for ev in events:
        i     = ev["chunk"]
        cands = ev["tts_candidates"]

        # ── playback ──────────────────────────────────────────────────────────
        dur = ev["play_end"] - ev["play_start"]
        ax.barh(Y_PLAY, dur, left=ev["play_start"], height=BAR_H,
                color=COLORS["play"], alpha=0.88, edgecolor="white", linewidth=0.6)
        ax.text(ev["play_start"] + dur / 2, Y_PLAY,
                f"▶{i}  {dur:.1f}s",
                ha="center", va="center", fontsize=8, fontweight="bold", color="white")

        # ── gap / silence ──────────────────────────────────────────────────────
        if ev["gap"] > 0.05:
            gap_left = ev["play_start"] - ev["gap"]
            ax.barh(Y_PLAY, ev["gap"], left=gap_left, height=BAR_H,
                    color=COLORS["gap"], alpha=0.80, edgecolor="white", linewidth=0.6)
            ax.text(gap_left + ev["gap"] / 2, Y_PLAY,
                    f"⚠ {ev['gap']:.1f}s",
                    ha="center", va="center", fontsize=7, color="white")

        # ── TTS candidates (parallel band) ────────────────────────────────────
        prep_end = ev["prep_start"] + rows[i]["total_elapsed_s"]
        for cand in cands:
            y = _tts_y(cand["k"])
            if cand["duration"] > 0:
                color = COLORS["tts_ok"] if cand["is_chosen"] else COLORS["tts_bg"]
                ec    = "black"  if cand["is_chosen"] else "white"
                lw    = 1.4     if cand["is_chosen"] else 0.4
                ax.barh(y, cand["duration"], left=cand["submit_t"],
                        height=TTS_H, color=color, alpha=0.90,
                        edgecolor=ec, linewidth=lw)
                label = f"{'★' if cand['is_chosen'] else ''}T{cand['k']} {cand['duration']:.1f}s"
                ax.text(cand["submit_t"] + cand["duration"] / 2, y,
                        label, ha="center", va="center", fontsize=6.5,
                        color="white" if cand["is_chosen"] else "#333333")
            else:
                # still running at selection: dashed arrow to prep_end
                ax.annotate(
                    "", xy=(prep_end, y), xytext=(cand["submit_t"], y),
                    arrowprops=dict(arrowstyle="->", color="#999999",
                                   lw=0.9, linestyle="dashed"),
                )
                ax.text(cand["submit_t"] + 0.2, y + TTS_H * 0.55,
                        f"T{cand['k']} (still running)",
                        fontsize=5.5, color="#888888", va="bottom")

        # ── main thread (fs / llm) ─────────────────────────────────────────────
        for seg_start, seg_end, kind in ev["main_segments"]:
            seg_dur = seg_end - seg_start
            if seg_dur < 0.05:
                continue
            ax.barh(Y_MAIN, seg_dur, left=seg_start, height=BAR_H,
                    color=COLORS[kind], alpha=0.88, edgecolor="white", linewidth=0.5)
            ax.text(seg_start + seg_dur / 2, Y_MAIN,
                    f"{kind}\n{seg_dur:.1f}s",
                    ha="center", va="center", fontsize=6.5, color="white")

    # chunk boundary lines at each play_start
    for ev in events:
        ax.axvline(ev["play_start"], color="gray", linewidth=0.9,
                   linestyle="--", alpha=0.35, zorder=0)

    # end-of-playback marker
    total_end = events[-1]["play_end"]
    ax.axvline(total_end, color="black", linewidth=1.8, zorder=3)
    ax.text(total_end, Y_PLAY + BAR_H / 2 + 0.1,
            f" ⏹ {total_end:.1f}s",
            ha="left", va="bottom", fontsize=9, fontweight="bold")

    # y-axis ticks
    tts_mid = _tts_y((max_cands - 1) / 2)
    ax.set_yticks([Y_MAIN, tts_mid, Y_PLAY])
    ax.set_yticklabels(["Main thread\n(fs + llm)", "TTS candidates\n(parallel)", "Playback"],
                       fontsize=9)
    ax.set_xlabel("Wall-clock time (s)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(Y_MAIN - 0.5, Y_PLAY + 0.7)
    ax.grid(axis="x", alpha=0.25)

    legend_patches = [
        mpatches.Patch(color=COLORS["play"],   label="Playback"),
        mpatches.Patch(color=COLORS["fs"],     label="FS estimation (main thread)"),
        mpatches.Patch(color=COLORS["llm"],    label="LLM refinement (main thread)"),
        mpatches.Patch(color=COLORS["tts_ok"], label="TTS chosen (★)"),
        mpatches.Patch(color=COLORS["tts_bg"], label="TTS done (not chosen)"),
        mpatches.Patch(color=COLORS["tts_nd"], label="TTS still running at selection"),
        mpatches.Patch(color=COLORS["gap"],    label="Gap / silence"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(rows, events):
    print(f"\n{'chunk':>6}  {'play_start':>10}  {'play_end':>8}  "
          f"{'gap_s':>6}  {'prep_s':>7}  {'budget_s':>8}  "
          f"{'cands':>5}  {'done':>4}  {'used':>4}  {'audio_s':>7}")
    print("-" * 90)
    for ev in events:
        i = ev["chunk"]
        r = rows[i]
        print(f"{i:>6}  {ev['play_start']:>10.2f}  {ev['play_end']:>8.2f}  "
              f"{ev['gap']:>6.2f}  "
              f"{r['total_elapsed_s']:>7.2f}  "
              f"{r['time_budget_s']:>8.2f}  "
              f"{int(r['n_candidates_submitted']):>5}  "
              f"{int(r['n_candidates_done']):>4}  "
              f"{int(r['used_candidate_iter']):>4}  "
              f"{r['audio_seconds']:>7.3f}")
    total_play = events[-1]["play_end"] - events[0]["play_start"]
    total_audio = sum(r["audio_seconds"] for r in rows)
    total_gap   = sum(ev["gap"] for ev in events)
    print("-" * 90)
    print(f"  Total playback span : {total_play:.2f}s")
    print(f"  Total audio content : {total_audio:.2f}s")
    print(f"  Total gap / silence : {total_gap:.2f}s")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to chunk_profile.csv (from 09.py)")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_path = (Path(args.output) if args.output
                else csv_path.parent / "overlap_timeline_par.png")

    rows   = load_profile(csv_path)
    events = compute_timeline(rows)

    title = f"Parallel Chunk Timeline — {csv_path.parent.name}"
    plot_timeline(rows, events, out_path, title)
    print_summary(rows, events)
