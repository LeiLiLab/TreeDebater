"""Summarize frozen replay results; bootstrap paired differences by case, not call."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sqlite3
import statistics


ROOT = Path(__file__).resolve().parent


def mean(xs):
    return statistics.mean(xs) if xs else None


def checklist(result):
    return mean([int(c["passed"]) for c in result["judge"]["checks"]])


def interval(differences):
    rng = random.Random(20261002)
    boot = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
    return [boot[249], boot[9749]]


def summarize(run_id):
    run = ROOT / "run" / run_id
    metadata_files = list(run.glob("metadata_worker*.json"))
    if not metadata_files:
        raise ValueError("No run metadata")
    metadata = json.loads(metadata_files[0].read_text())
    if any(json.loads(p.read_text()) != metadata for p in metadata_files):
        raise ValueError("Worker configurations differ")
    results = [json.loads(p.read_text()) for p in run.glob("*__*.json")]
    complete = [r for r in results if "judge" in r]
    expected = {(case, mode, rep) for case in metadata["case_ids"]
                for mode in metadata["modes"] for rep in range(metadata["repeats"])}
    observed = {(r["case"], r["mode"], r["repeat"]) for r in complete}
    if observed - expected:
        raise ValueError("Unexpected result identities")
    by_mode = defaultdict(list)
    for r in complete:
        by_mode[r["mode"]].append(r)
    rows = []
    for mode in metadata["modes"]:
        rs = by_mode[mode]
        if not rs:
            continue
        latencies = [r["estimated_residual_text_seconds"] for r in rs]
        rows.append({"mode": mode, "n": len(rs), "checklist_rate": mean([checklist(r) for r in rs]),
                     "strength_mean": mean([r["judge"]["rebuttal_strength"] for r in rs]),
                     "strawman_rate": mean([int(r["judge"]["strawman"]) for r in rs]),
                     "unsupported_fact_rate": mean([int(r["judge"]["unsupported_facts"]) for r in rs]),
                     "residual_text_mean_s": mean(latencies),
                     "residual_text_median_s": statistics.median(latencies),
                     "generation_calls_mean": mean([r["model_usage"]["calls"] for r in rs]),
                     "generation_cost_mean_usd": mean([r["model_usage"]["reported_usage_estimate_usd"] for r in rs]),
                     "answer_words_mean": mean([r["answer_words"] for r in rs]),
                     "gate_waits": sum(e["action"] == "WAIT" for r in rs for e in r["after_generation"]["events"]),
                     "revisions": sum(len(r["after_generation"]["opponent_tree"].get("revisions", [])) for r in rs)})
    pairs = {}
    # Average repeated runs within each case before estimating uncertainty.
    for mode in metadata["modes"]:
        if mode == "legacy":
            continue
        paired = []
        for case in metadata["case_ids"]:
            base = [r for r in by_mode["legacy"] if r["case"] == case]
            test = [r for r in by_mode[mode] if r["case"] == case]
            if len(base) != metadata["repeats"] or len(test) != metadata["repeats"]:
                continue
            paired.append({"case": case,
                           "quality_diff": mean([checklist(r) for r in test])-mean([checklist(r) for r in base]),
                           "latency_diff_s": mean([r["estimated_residual_text_seconds"] for r in test])
                                             -mean([r["estimated_residual_text_seconds"] for r in base])})
        if paired:
            pairs[mode] = {"case_count": len(paired), "case_differences": paired,
                           "quality_diff_mean": mean([p["quality_diff"] for p in paired]),
                           "quality_diff_95ci": interval([p["quality_diff"] for p in paired]),
                           "latency_diff_mean_s": mean([p["latency_diff_s"] for p in paired]),
                           "latency_diff_95ci_s": interval([p["latency_diff_s"] for p in paired])}
    db = sqlite3.connect(ROOT / "run/cost.sqlite")
    # IDs are validated by the runner. A prefix comparison avoids LIKE wildcards.
    calls = [r for r in db.execute("SELECT label,reserved,state,input_tokens,output_tokens,estimated_usd FROM calls")
             if r[0].startswith(run_id + "/")]
    usage = {"calls": len(calls), "reserved_upper_usd": sum(c[1] for c in calls),
             "unresolved_calls": sum(c[2] != "ok" for c in calls),
             "input_tokens": sum(c[3] or 0 for c in calls), "output_tokens": sum(c[4] or 0 for c in calls),
             "reported_usage_estimate_usd": sum(c[5] or 0 for c in calls)}
    summary = {"run_id": run_id, "metadata": metadata, "expected_answers": len(expected),
               "completed_answers": len(observed), "missing": sorted(expected-observed),
               "metrics": rows, "paired_vs_legacy": pairs, "usage_including_judging": usage,
               "limitations": ["Small authored scenario set, not a standard debate benchmark",
                               "Same model generates and judges; no human or independent-model validation",
                               "Text-only replay; latency uses a simulated schedule and excludes ASR/TTS/network playback",
                               "Exact-target matching shared across arms; embedding fallback disabled",
                               "Cost uses provider-reported tokens and published rates, not a settled invoice"]}
    (ROOT / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{run_id}: {len(observed)}/{len(expected)} answers")
    print("mode\tn\tchecklist\tstrength\tstrawman\tresidual_mean_s\tcalls\tcost_usd")
    for r in rows:
        print(f"{r['mode']}\t{r['n']}\t{r['checklist_rate']:.3f}\t{r['strength_mean']:.2f}\t"
              f"{r['strawman_rate']:.3f}\t{r['residual_text_mean_s']:.2f}\t"
              f"{r['generation_calls_mean']:.2f}\t{r['generation_cost_mean_usd']:.6f}")
    print(json.dumps(usage))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    summarize(ap.parse_args().run_id)
