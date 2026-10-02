"""Resume frozen answers; allow one documented 1600-token recovery per failed judge.

Primary judging stays at 800 tokens. Completed judgments are never replaced.
Only truncated/error prior judge calls trigger the larger recovery allowance.
All requests retain the same shared budget, model, prompt and decoding settings.
"""
import json
from pathlib import Path

from scripts import benchmark_incremental_planning as benchmark

original_judge = benchmark.judge


def recovery_judge(case, answer, client, model):
    artifacts = []
    for (request_id,) in client.db.execute("SELECT id FROM calls WHERE label=?", (client.label,)):
        path = client.directory / f"call_{request_id:06}.json"
        if path.exists():
            item = json.loads(path.read_text())
            if item["request"]["model"] == model:
                artifacts.append(item)
    if any(a["request"]["max_tokens"] == 1600 for a in artifacts):
        raise RuntimeError("A recovery judge was already attempted; refusing unlimited retries")
    failed = any(a.get("error") or a.get("truncated") for a in artifacts)
    if not failed:
        try:
            return original_judge(case, answer, client, model)
        except (ValueError, KeyError) as exc:
            print("JUDGE_RECOVERY", client.label, type(exc).__name__, flush=True)
    complete = client.complete
    def enlarged(*args, **kwargs):
        if kwargs.get("model") == model:
            kwargs["max_tokens"] = 1600
        return complete(*args, **kwargs)
    client.complete = enlarged
    try:
        result = original_judge(case, answer, client, model)
        result["recovery_output_cap"] = 1600
        return result
    finally:
        client.complete = complete


if __name__ == "__main__":
    benchmark.judge = recovery_judge
    benchmark.main()
