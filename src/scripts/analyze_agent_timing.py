#!/usr/bin/env python3
"""
Analyze agent / LLM timing lines from TreeDebater main logs (``[timing]``) and optional I/O logs (``[io]``).

Parses the format emitted by ``utils/timing_log.py`` (see ``src/scripts/README_LOGGING.md``, Agent section).

Usage:
    python src/scripts/analyze_agent_timing.py log_files/14.log
    python src/scripts/analyze_agent_timing.py log_files/14.log --io-log log_files/14_io.log
    python src/scripts/analyze_agent_timing.py log_files/14.log --json-out report.json
    python src/scripts/analyze_agent_timing.py log_files/38.log --include-tts
    python src/scripts/analyze_agent_timing.py log_files/38.log --include-phases baseline_api_http
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


# Strip standard debate file formatter prefix: "... DEBUG module - funcName: message"
_PREFIX_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\w+\s+[\w.]+\s+-\s+\w+:\s*"
)

_TIMING_HEAD = re.compile(
    r"^\[timing\]\s+phase=(\S+)\s+duration_s=([\d.]+)\s*(.*)$"
)
_TIMING_META = re.compile(
    r"^\[timing-meta\]\s+(.*)$"
)
_TURN_RE = re.compile(
    r"\[Turn\]\s+turn_(start|end)\s+stage=(\S+)\s+side=(\S+)\s+t=([\d.]+)"
)
_KV = re.compile(r"(\w+)=([^\s]+)")

# Phases to exclude from analysis output/report.
# Edit this list to hide noisy phases without changing runtime logging.
EXCLUDED_PHASES = {
    "tts_wall_clock",
    "length_adjust",
}

# Phases grouped for the human-readable report (edit as you add new phases)
MACRO_PHASES = frozenset(
    {
        "env_stage_wall",
        "evaluation_wall",
        "comparison_phase_wall",
        "comparison_evaluation_total_wall",
        "compare_env_stage_wall",
        "prepare_claim_pool_wall",
        "io_log_ready",
    }
)
SPEAK_PIPELINE = frozenset(
    {
        "tree_debater_speak",
        "main_get_response",
        "revision_suggestion",
        "length_adjust",
        "length_adjust_iteration",
        "post_process",
    }
)
LISTENER_TREE = frozenset(
    {
        "listen_analyze_statement",
        "analyze_statement",
    }
)
AUDIENCE_REVISION = frozenset(
    {
        "audience_exemplar_retrieval",
        "audience_simulated_feedback_llm",
        "evidence_selection_llm",
    }
)
RETRIEVAL = frozenset(
    {
        "rehearsal_retrieve_on_prepared_tree",
        "exemplar_retrieval_query_embedding",
        "exemplar_retrieval_semantic_search",
    }
)
ATOM_LLM = frozenset(
    {
        "helper_client_litellm",
        "debater_litellm_completion",
        "get_response_with_retry_llm",
    }
)
ATOM_OTHER = frozenset(
    {
        "embedding_api_fetch",
        "tts_wall_clock",
        "tts_trim_wall_clock",
    }
)
BASELINE_API = frozenset({"baseline_api_http"})


@dataclass
class TimingRecord:
    phase: str
    duration_s: float
    fields: Dict[str, str] = field(default_factory=dict)
    raw: str = ""


@dataclass
class TurnWall:
    stage: str
    side: str
    start_t: Optional[float] = None
    end_t: Optional[float] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.start_t is not None and self.end_t is not None:
            return self.end_t - self.start_t
        return None


def _parse_kv_tail(tail: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in _KV.finditer(tail.strip()):
        out[m.group(1)] = m.group(2)
    return out


def _strip_log_prefix(line: str) -> str:
    line = line.rstrip("\n")
    m = _PREFIX_RE.match(line)
    if m:
        return line[m.end() :]
    return line


def parse_timing_line(line: str) -> Optional[TimingRecord]:
    s = _strip_log_prefix(line)
    if not s.startswith("[timing]"):
        return None
    m = _TIMING_HEAD.match(s)
    if not m:
        return None
    phase, dur_s, tail = m.group(1), float(m.group(2)), m.group(3) or ""
    return TimingRecord(phase=phase, duration_s=dur_s, fields=_parse_kv_tail(tail), raw=s)


def parse_timing_meta(line: str) -> Optional[Dict[str, str]]:
    s = _strip_log_prefix(line)
    if not s.startswith("[timing-meta]"):
        return None
    m = _TIMING_META.match(s)
    if not m:
        return None
    return _parse_kv_tail(m.group(1))


def parse_io_header_line(line: str) -> Optional[Dict[str, str]]:
    s = _strip_log_prefix(line)
    if "[io]" not in s:
        return None
    # First line of a block: "[io] call_id=1 phase=... title=..."
    idx = s.find("[io]")
    if idx < 0:
        return None
    rest = s[idx + len("[io]") :].strip()
    return _parse_kv_tail(rest)


def load_timing_records(path: Path) -> List[TimingRecord]:
    records: List[TimingRecord] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = parse_timing_line(line)
            if rec:
                records.append(rec)
    return records


def load_turn_walls(path: Path) -> Dict[Tuple[str, str], TurnWall]:
    turns: Dict[Tuple[str, str], TurnWall] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = _strip_log_prefix(line)
            m = _TURN_RE.search(s)
            if not m:
                continue
            event, stage, side, ts_s = m.group(1), m.group(2), m.group(3), float(m.group(4))
            key = (stage, side)
            if key not in turns:
                turns[key] = TurnWall(stage=stage, side=side)
            if event == "start":
                turns[key].start_t = ts_s
            else:
                turns[key].end_t = ts_s
    return turns


def sum_by_phase_stage_side(
    records: List[TimingRecord], phase: str
) -> Dict[Tuple[str, str], float]:
    out: DefaultDict[Tuple[str, str], float] = defaultdict(float)
    for r in records:
        if r.phase != phase:
            continue
        key = (r.fields.get("stage", "?"), r.fields.get("side", "?"))
        out[key] += r.duration_s
    return dict(out)


def sum_response_cost(records: List[TimingRecord]) -> Tuple[float, int]:
    total, n = 0.0, 0
    for r in records:
        cost_s = r.fields.get("response_cost")
        if cost_s is None:
            continue
        try:
            total += float(cost_s)
            n += 1
        except ValueError:
            continue
    return total, n


def build_excluded_phases(
    base: set[str], include_phases: Optional[List[str]], include_tts: bool
) -> set[str]:
    excluded = set(base)
    if include_tts:
        excluded.discard("tts_wall_clock")
    if include_phases:
        for ph in include_phases:
            for part in ph.split(","):
                part = part.strip()
                if part:
                    excluded.discard(part)
    return excluded


def filter_excluded_phases(
    records: List[TimingRecord], excluded_phases: set[str]
) -> Tuple[List[TimingRecord], int]:
    if not excluded_phases:
        return records, 0
    filtered = [r for r in records if r.phase not in excluded_phases]
    return filtered, len(records) - len(filtered)


def load_meta_records(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = parse_timing_meta(line)
            if m:
                rows.append(m)
    return rows


def count_io_blocks(path: Path) -> Tuple[int, Dict[Tuple[str, str], int]]:
    """Count I/O blocks (header lines with ``[io]``) and histogram (phase, title)."""
    total = 0
    hist: Dict[Tuple[str, str], int] = defaultdict(int)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            h = parse_io_header_line(line)
            if h and "call_id" in h:
                total += 1
                key = (h.get("phase", "?"), h.get("title", "?"))
                hist[key] += 1
    return total, dict(hist)


def aggregate_by_phase(records: List[TimingRecord]) -> Dict[str, Dict[str, Any]]:
    by_phase: DefaultDict[str, List[float]] = defaultdict(list)
    for r in records:
        by_phase[r.phase].append(r.duration_s)
    out: Dict[str, Dict[str, Any]] = {}
    for phase, xs in sorted(by_phase.items()):
        out[phase] = {
            "count": len(xs),
            "total_s": round(sum(xs), 4),
            "mean_s": round(sum(xs) / len(xs), 4),
            "min_s": round(min(xs), 4),
            "max_s": round(max(xs), 4),
        }
    return out


def group_by_call_id(records: List[TimingRecord]) -> Dict[str, List[TimingRecord]]:
    groups: DefaultDict[str, List[TimingRecord]] = defaultdict(list)
    for r in records:
        cid = r.fields.get("call_id")
        if cid is not None:
            groups[cid].append(r)
    return dict(groups)


def print_report(
    records: List[TimingRecord],
    records_all: List[TimingRecord],
    meta: List[Dict[str, str]],
    io_path: Optional[Path],
    verbose: bool,
    turns: Dict[Tuple[str, str], TurnWall],
    excluded_phases: Optional[set[str]] = None,
    filtered_count: int = 0,
) -> Dict[str, Any]:
    agg = aggregate_by_phase(records)
    by_cid = group_by_call_id(records)

    lines: List[str] = []
    W = lines.append

    W("=" * 80)
    W("AGENT / LLM TIMING ANALYSIS ([timing] lines)")
    W("=" * 80)
    W(f"Total timing records: {len(records)}")
    if excluded_phases:
        W(f"Excluded phases: {sorted(excluded_phases)}")
        W(f"Excluded records: {filtered_count}")
    W("")

    # --- Bucket summary ---
    def bucket_sum(phases: frozenset) -> Tuple[int, float]:
        n, t = 0, 0.0
        for p in phases:
            if p in agg:
                n += agg[p]["count"]
                t += agg[p]["total_s"]
        return n, t

    W("--- Interest summary (by category) ---")
    for name, pset in [
        ("Macro (env / eval / compare / prep)", MACRO_PHASES),
        ("Baseline API (agent4debate HTTP)", BASELINE_API),
        ("Speak pipeline (TreeDebater turn)", SPEAK_PIPELINE),
        ("Listen + debate-flow tree", LISTENER_TREE),
        ("Audience + revision LLM blocks", AUDIENCE_REVISION),
        ("Retrieval (exemplar + rehearsal)", RETRIEVAL),
        ("Atom: LLM completions", ATOM_LLM),
        ("Atom: embed / TTS wall", ATOM_OTHER),
    ]:
        n, t = bucket_sum(pset)
        W(f"  {name}: events={n}  total_time_s={t:.2f}")
    W("")

    cost_total, cost_n = sum_response_cost(records_all)
    if cost_n:
        W("--- LLM response cost (response_cost= on [timing] lines) ---")
        W(f"  total_usd={cost_total:.4f}  events_with_cost={cost_n}")
        W("")

    # --- Turn wall clock + per-turn breakdown ---
    if turns:
        api_by = sum_by_phase_stage_side(records_all, "baseline_api_http")
        tts_by = sum_by_phase_stage_side(records_all, "tts_wall_clock")
        trim_by = sum_by_phase_stage_side(records_all, "tts_trim_wall_clock")
        W("--- Debate turn wall clock ([Turn] lines) ---")
        W(f"{'stage':<10} {'side':<8} {'turn_s':>10} {'api_s':>10} {'tts_s':>10} {'trim_s':>10} {'other_s':>10}")
        turn_total = 0.0
        api_total = tts_total = trim_total = 0.0
        stage_order = ["opening", "rebuttal", "closing"]
        side_order = ["for", "against"]

        def _sort_key(item: Tuple[Tuple[str, str], TurnWall]) -> Tuple[int, int]:
            (stage, side), _ = item
            si = stage_order.index(stage) if stage in stage_order else 99
            si2 = side_order.index(side) if side in side_order else 99
            return si, si2

        for (stage, side), tw in sorted(turns.items(), key=_sort_key):
            dur = tw.duration_s
            if dur is None:
                continue
            api_s = api_by.get((stage, side), 0.0)
            tts_s = tts_by.get((stage, side), 0.0)
            trim_s = trim_by.get((stage, side), 0.0)
            other_s = max(0.0, dur - api_s - tts_s - trim_s)
            turn_total += dur
            api_total += api_s
            tts_total += tts_s
            trim_total += trim_s
            W(
                f"{stage:<10} {side:<8} {dur:10.2f} {api_s:10.2f} {tts_s:10.2f} {trim_s:10.2f} {other_s:10.2f}"
            )
        W(
            f"{'TOTAL':<10} {'':<8} {turn_total:10.2f} {api_total:10.2f} {tts_total:10.2f} {trim_total:10.2f} "
            f"{max(0.0, turn_total - api_total - tts_total - trim_total):10.2f}"
        )
        if not api_by and turn_total > 0:
            W("  (baseline_api_http not logged — re-run debate after instrumentation, or use turn_s as upper bound)")
        W("")

    # --- Per-phase table ---
    W("--- Per-phase statistics ---")
    W(f"{'phase':<42} {'n':>5} {'total_s':>10} {'mean_s':>10} {'max_s':>10}")
    for phase in sorted(agg.keys()):
        a = agg[phase]
        W(f"{phase:<42} {a['count']:>5} {a['total_s']:>10.2f} {a['mean_s']:>10.2f} {a['max_s']:>10.2f}")
    W("")

    # --- Speak sessions by call_id ---
    if by_cid:
        W("--- TreeDebater speak sessions (by call_id) ---")
        for cid in sorted(by_cid.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            sess = by_cid[cid]
            total = sum(r.duration_s for r in sess if r.phase == "tree_debater_speak")
            W(f"  call_id={cid}  (tree_debater_speak wall={total:.2f}s if present)")
            for r in sess:
                if r.phase == "tree_debater_speak":
                    continue
                extra = " ".join(f"{k}={v}" for k, v in sorted(r.fields.items()) if k != "call_id")
                W(f"    {r.duration_s:8.2f}s  {r.phase}" + (f"  | {extra}" if extra else ""))
            W("")
    else:
        W("--- No records with call_id= (speak pipeline grouping skipped) ---")
        W("")

    # --- Length-adjust iterations ---
    iters = [r for r in records if r.phase == "length_adjust_iteration"]
    if iters:
        W("--- Length adjust iterations ---")
        for r in iters:
            W(
                f"  iter={r.fields.get('iteration', '?')} max_retry={r.fields.get('max_retry', '?')} "
                f"fit_ok={r.fields.get('fit_ok', '?')} cost={r.fields.get('current_cost', '?')} "
                f"duration_s={r.duration_s:.3f} stage={r.fields.get('stage')} side={r.fields.get('side')}"
            )
        W("")

    # --- Slowest single events ---
    W("--- Slowest 25 timing events ---")
    slow = sorted(records, key=lambda r: r.duration_s, reverse=True)[:25]
    for r in slow:
        loc = f"{r.fields.get('stage', '')}/{r.fields.get('side', '')}".strip("/")
        W(f"  {r.duration_s:10.2f}s  {r.phase}" + (f"  ({loc})" if loc else ""))
    W("")

    # --- Meta lines ---
    if meta:
        W(f"--- timing-meta lines: {len(meta)} ---")
        if verbose:
            for m in meta[:50]:
                W(f"  {m}")
            if len(meta) > 50:
                W(f"  ... ({len(meta) - 50} more)")
        W("")

    # --- I/O log ---
    io_report: Dict[str, Any] = {}
    if io_path and io_path.is_file():
        n_io, hist = count_io_blocks(io_path)
        W("--- I/O log (prompt/response blocks) ---")
        W(f"  file: {io_path}")
        W(f"  total [io] blocks: {n_io}")
        if hist and verbose:
            W("  histogram (phase, title) -> count:")
            for (ph, title), c in sorted(hist.items(), key=lambda x: -x[1])[:40]:
                W(f"    ({ph}, {title}): {c}")
        W("")
        io_report = {"io_file": str(io_path), "io_blocks": n_io, "histogram": {f"{a}|{b}": c for (a, b), c in hist.items()}}
    elif io_path:
        W(f"--- I/O log not found: {io_path} ---")
        W("")

    text = "\n".join(lines)
    print(text)

    turn_report: Dict[str, Any] = {}
    if turns:
        api_all = sum_by_phase_stage_side(records_all, "baseline_api_http")
        tts_all = sum_by_phase_stage_side(records_all, "tts_wall_clock")
        trim_all = sum_by_phase_stage_side(records_all, "tts_trim_wall_clock")
        rows = []
        for (stage, side), tw in sorted(turns.items()):
            if tw.duration_s is None:
                continue
            rows.append(
                {
                    "stage": stage,
                    "side": side,
                    "turn_wall_s": round(tw.duration_s, 4),
                    "baseline_api_s": round(api_all.get((stage, side), 0.0), 4),
                    "tts_wall_s": round(tts_all.get((stage, side), 0.0), 4),
                    "tts_trim_s": round(trim_all.get((stage, side), 0.0), 4),
                }
            )
        turn_report = {"turns": rows, "turn_wall_total_s": round(sum(r["turn_wall_s"] for r in rows), 4)}

    return {
        "total_records": len(records),
        "excluded_phases": sorted(excluded_phases) if excluded_phases else [],
        "excluded_records": filtered_count,
        "by_phase": agg,
        "call_id_sessions": {k: [asdict(x) for x in v] for k, v in by_cid.items()},
        "meta_count": len(meta),
        "io": io_report,
        "response_cost_usd": round(cost_total, 6) if cost_n else None,
        "response_cost_events": cost_n,
        "turns": turn_report,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze [timing] / [io] agent logs.")
    ap.add_argument("log_file", type=Path, help="Main debate log (e.g. log_files/14.log)")
    ap.add_argument("--io-log", type=Path, default=None, help="I/O log path (default: <N>_io.log next to main log)")
    ap.add_argument("--json-out", type=Path, default=None, help="Write structured JSON summary")
    ap.add_argument("--verbose", "-v", action="store_true", help="Extra detail (meta + I/O histogram)")
    ap.add_argument(
        "--include-phases",
        action="append",
        default=[],
        metavar="PHASE",
        help="Include these phases in stats (comma-separated ok). Repeatable.",
    )
    ap.add_argument(
        "--include-tts",
        action="store_true",
        help="Include tts_wall_clock in per-phase stats (excluded by default).",
    )
    args = ap.parse_args()

    main_log = args.log_file
    if not main_log.is_file():
        raise SystemExit(f"Log not found: {main_log}")

    io_log = args.io_log
    if io_log is None:
        p = str(main_log)
        if p.endswith(".log"):
            candidate = Path(p.replace(".log", "_io.log"))
            if candidate.is_file():
                io_log = candidate

    records_raw = load_timing_records(main_log)
    excluded = build_excluded_phases(EXCLUDED_PHASES, args.include_phases, args.include_tts)
    records, filtered_count = filter_excluded_phases(records_raw, excluded)
    meta = load_meta_records(main_log)
    turns = load_turn_walls(main_log)

    report = print_report(
        records,
        records_raw,
        meta,
        io_log,
        args.verbose,
        turns,
        excluded_phases=excluded,
        filtered_count=filtered_count,
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
