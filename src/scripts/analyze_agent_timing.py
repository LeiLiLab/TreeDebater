#!/usr/bin/env python3
"""
Analyze agent / LLM timing lines from TreeDebater main logs (``[timing]``) and optional I/O logs (``[io]``).

Parses the format emitted by ``utils/timing_log.py`` (see ``src/scripts/README_LOGGING.md``, Agent section).

Usage:
    python src/scripts/analyze_agent_timing.py log_files/14.log
    python src/scripts/analyze_agent_timing.py log_files/14.log --io-log log_files/14_io.log
    python src/scripts/analyze_agent_timing.py log_files/14.log --json-out report.json
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
_KV = re.compile(r"(\w+)=([^\s]+)")

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


@dataclass
class TimingRecord:
    phase: str
    duration_s: float
    fields: Dict[str, str] = field(default_factory=dict)
    raw: str = ""


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
    meta: List[Dict[str, str]],
    io_path: Optional[Path],
    verbose: bool,
) -> Dict[str, Any]:
    agg = aggregate_by_phase(records)
    by_cid = group_by_call_id(records)

    lines: List[str] = []
    W = lines.append

    W("=" * 80)
    W("AGENT / LLM TIMING ANALYSIS ([timing] lines)")
    W("=" * 80)
    W(f"Total timing records: {len(records)}")
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

    return {
        "total_records": len(records),
        "by_phase": agg,
        "call_id_sessions": {k: [asdict(x) for x in v] for k, v in by_cid.items()},
        "meta_count": len(meta),
        "io": io_report,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze [timing] / [io] agent logs.")
    ap.add_argument("log_file", type=Path, help="Main debate log (e.g. log_files/14.log)")
    ap.add_argument("--io-log", type=Path, default=None, help="I/O log path (default: <N>_io.log next to main log)")
    ap.add_argument("--json-out", type=Path, default=None, help="Write structured JSON summary")
    ap.add_argument("--verbose", "-v", action="store_true", help="Extra detail (meta + I/O histogram)")
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

    records = load_timing_records(main_log)
    meta = load_meta_records(main_log)

    report = print_report(records, meta, io_log, args.verbose)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
