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


def paired_comparison(by_mode, metadata, baseline, candidate):
    paired = []
    for case in metadata["case_ids"]:
        base = [r for r in by_mode[baseline] if r["case"] == case]
        test = [r for r in by_mode[candidate] if r["case"] == case]
        if len(base) != metadata["repeats"] or len(test) != metadata["repeats"]:
            continue
        paired.append({"case": case,
                       "quality_diff": mean([checklist(r) for r in test])-mean([checklist(r) for r in base]),
                       "latency_diff_s": mean([r["estimated_residual_text_seconds"] for r in test])
                                         -mean([r["estimated_residual_text_seconds"] for r in base])})
    if not paired:
        return None
    return {"baseline": baseline, "candidate": candidate,
            "case_count": len(paired), "case_differences": paired,
            "quality_diff_mean": mean([p["quality_diff"] for p in paired]),
            "quality_diff_95ci": interval([p["quality_diff"] for p in paired]),
            "latency_diff_mean_s": mean([p["latency_diff_s"] for p in paired]),
            "latency_diff_95ci_s": interval([p["latency_diff_s"] for p in paired])}


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
    reference = next((m for m in ("legacy", "linear", "tree_plan") if m in metadata["modes"]),
                     metadata["modes"][0])
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
                     "residual_worker_return_mean_s": mean([
                         r["remaining_preparation_seconds"] + r["full_generation_including_own_analysis_seconds"]
                         for r in rs]),
                     "generation_calls_mean": mean([r["model_usage"]["calls"] for r in rs]),
                     "prior_context_setup_calls_mean": mean([r.get("prior_context_setup_calls", 0) for r in rs]),
                     "live_generation_calls_mean": mean([r.get("live_generation_calls", r["model_usage"]["calls"]) for r in rs]),
                     "generation_cost_mean_usd": mean([r["model_usage"]["reported_usage_estimate_usd"] for r in rs]),
                     "answer_words_mean": mean([r["answer_words"] for r in rs]),
                     "gate_waits": sum(e["action"] == "WAIT" for r in rs for e in r["after_generation"]["events"]),
                     "duplicate_skips": sum(e["action"] == "SKIP_DUPLICATE" for r in rs for e in r["after_generation"]["events"]),
                     "incomplete_waits": sum(e["action"] == "WAIT_INCOMPLETE" for r in rs for e in r["after_generation"]["events"]),
                     "invalid_states": sum(e["action"] == "INVALID_STATE" for r in rs for e in r["after_generation"]["events"]),
                     "invalid_targets": sum(e["action"] == "INVALID_TARGET" for r in rs for e in r["after_generation"]["events"]),
                     "revisions": sum(len(r["after_generation"]["opponent_tree"].get("revisions", [])) for r in rs)})
    pairs = {}
    # Average repeated runs within each case before estimating uncertainty.
    for mode in metadata["modes"]:
        if mode == reference:
            continue
        paired = paired_comparison(by_mode, metadata, reference, mode)
        if paired:
            pairs[mode] = paired
    components = {}
    for baseline, candidate in (("linear", "adaptive_linear"),
                                ("corrected_tree", "tree_plan"),
                                ("tree_plan", "adaptive_tree"),
                                ("linear", "structured_linear"),
                                ("structured_linear", "grounded_linear"),
                                ("grounded_linear", "light_linear"),
                                ("tree_plan", "grounded_tree"),
                                ("grounded_tree", "light_tree"),
                                ("grounded_linear", "grounded_tree"),
                                ("light_linear", "light_tree")):
        paired = paired_comparison(by_mode, metadata, baseline, candidate)
        if paired:
            components[candidate + "_vs_" + baseline] = paired
    db = sqlite3.connect(ROOT / "run/cost.sqlite")
    # IDs are validated by the runner. A prefix comparison avoids LIKE wildcards.
    calls = [r for r in db.execute("SELECT label,reserved,state,input_tokens,output_tokens,estimated_usd FROM calls")
             if r[0].startswith(run_id + "/")]
    usage = {"calls": len(calls), "reserved_upper_usd": sum(c[1] for c in calls),
             "unresolved_calls": sum(c[2] != "ok" for c in calls),
             "input_tokens": sum(c[3] or 0 for c in calls), "output_tokens": sum(c[4] or 0 for c in calls),
             "reported_usage_estimate_usd": sum(c[5] or 0 for c in calls)}
    usage["accounted_exposure_usd"] = sum(charge for label, charge in db.execute(
        """SELECT c.label, CASE WHEN c.state='ok' AND c.reserved=s.original_reserved
        THEN coalesce(s.charge_usd,c.reserved) ELSE c.reserved END
        FROM calls c LEFT JOIN budget_settlements s ON c.id=s.call_id""")
        if label.startswith(run_id + "/"))
    request_audit = {"missing_artifacts": [], "truncated_calls": [], "errors": []}
    for request_id, label in db.execute("SELECT id,label FROM calls"):
        if not label.startswith(run_id + "/"):
            continue
        artifact_path = ROOT / "run" / f"call_{request_id:06}.json"
        if not artifact_path.exists():
            request_audit["missing_artifacts"].append(request_id)
            continue
        artifact = json.loads(artifact_path.read_text())
        if artifact.get("truncated"):
            request_audit["truncated_calls"].append({"id": request_id, "label": label,
                                                    "max_tokens": artifact["request"]["max_tokens"]})
        if artifact.get("error"):
            request_audit["errors"].append({"id": request_id, "label": label,
                                           "error": artifact["error"]})
    warnings = defaultdict(int)
    for log_path in (ROOT / "run").glob(run_id + "-worker*.log"):
        current_label = "unknown"
        for line in log_path.read_text().splitlines():
            if line.startswith("START "):
                current_label = line[6:]
            if " WARNING " in line:
                message = line.split(" WARNING ", 1)[1]
                category = message.split(", target:", 1)[0]
                mode = current_label.split("/")[2] if current_label.count("/") >= 2 else "unknown"
                warnings[(mode, category)] += 1
    request_audit["runtime_warning_counts"] = [
        {"mode": mode, "warning": category, "count": count}
        for (mode, category), count in sorted(warnings.items())]
    summary = {"run_id": run_id, "metadata": metadata, "expected_answers": len(expected),
               "completed_answers": len(observed), "missing": sorted(expected-observed),
               "metrics": rows, "reference_mode": reference, "paired_vs_baseline": pairs,
               "paired_vs_legacy": pairs if reference == "legacy" else {}, "component_comparisons": components,
               "usage_including_judging": usage, "request_audit": request_audit,
               "limitations": ["Small authored scenario set, not a standard debate benchmark",
                               ("Same model generates and judges; no human or independent-model validation"
                                if metadata.get("judge_model", metadata["model"]) == metadata["model"]
                                else "Different generator/judge models, but no human adjudication; automatic judging remains fallible"),
                               "Spot checks found inconsistent constraint judgments; automated scores are provisional, not established accuracy",
                               "Zero unsupported-fact flags do not establish factual correctness",
                               "Text-only replay; latency uses a simulated schedule and excludes ASR/TTS/network playback",
                               "Text-ready timing precedes post-speech tree analysis; worker-return timing is reported separately",
                               "Exact-target matching shared across arms; embedding fallback disabled",
                               "Cost uses provider-reported tokens and published rates, not a settled invoice"]}
    recovered = [{"case": r["case"], "mode": r["mode"],
                  "output_cap": r["judge"]["recovery_output_cap"]}
                 for r in complete if "recovery_output_cap" in r["judge"]]
    if recovered:
        summary["judge_recovery"] = {
            "primary_output_cap": 800, "recovered_answers": recovered,
            "policy": "Successful primary judgments retained; one larger-cap recovery allowed after error/truncation, with unchanged prompt/model/answer."}
        summary["limitations"].append(
            "Some missing judgments required a 1600-token recovery after 800-token failures; judge output caps were not homogeneous.")
    summary["completion_audit"] = {
        "worker_completion_markers": len(list(run.glob("complete_worker*.json"))),
        "pending_requests": sum(c[2] == "pending" for c in calls),
        "failed_requests": sum(c[2] == "error" for c in calls)}
    diagnostics_path = ROOT / f"{run_id}_diagnostics.json"
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text())
        bindings = diagnostics["final_binding_checks"]
        fallback_count = sum(b["final_raw_prefix_fallback"] for b in bindings)
        summary["post_run_diagnostics"] = str(diagnostics_path.relative_to(ROOT))
        summary["limitations"].extend([
            "Composite checklist items can fail for omitted qualifications even when the answer does not contradict them.",
            f"{fallback_count}/{len(bindings)} grounded/light-tree answers fall back to the raw prefix; aggregate gains do not isolate tree binding from grounding feedback."])
        summary["limitations"].extend(diagnostics["limitations"])
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
