"""OpenAI-compatible experiment client with durable pre-dispatch cost reservations.

No automatic HTTP retries. Reserve before dispatch; after a successful response,
reconcile verified usage at a 4x cost margin. Failures/unknown usage keep their full
reservation. Original reservations and append-only settlements remain auditable.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import time
from urllib.request import Request, urlopen

from .experiment_accounting import (MODEL_RATES, accounted_exposure, initialize_accounting,
                                    reconcile_success)


MODEL = "google.gemma-4-26b-a4b"


class BudgetExceeded(RuntimeError):
    pass


class BudgetedClient:
    def __init__(self, directory, *, cap=200.0, base_url="http://127.0.0.1:4000/v1", label="pilot"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.base_url, self.label = base_url.rstrip("/"), label
        self.db = sqlite3.connect(self.directory / "cost.sqlite", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS budget(id INTEGER PRIMARY KEY, cap REAL NOT NULL)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS calls(
            id INTEGER PRIMARY KEY, label TEXT, created TEXT, reserved REAL, state TEXT,
            input_tokens INTEGER, output_tokens INTEGER, estimated_usd REAL, seconds REAL)""")
        self.db.execute("INSERT OR IGNORE INTO budget VALUES(1, ?)", (cap,))
        self.db.commit()
        if self.db.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0] != cap:
            raise ValueError("Cannot silently change an existing experiment budget")
        initialize_accounting(self.db)

    def summary(self, label=None):
        where, args = (" WHERE label=?", (label,)) if label is not None else ("", ())
        row = self.db.execute("""SELECT count(*), coalesce(sum(reserved),0),
            coalesce(sum(estimated_usd),0), sum(CASE WHEN state!='ok' THEN 1 ELSE 0 END),
                    coalesce(sum(input_tokens),0), coalesce(sum(output_tokens),0) FROM calls""" + where, args).fetchone()
        exposure = accounted_exposure(self.db, label)
        return dict(calls=row[0], reserved_upper_usd=row[1], reported_usage_estimate_usd=row[2],
                    # reserved_upper_usd is the historical sum, not current budget occupancy.
                    accounted_exposure_usd=exposure,
                    uncertain_calls=row[3] or 0, input_tokens=row[4], output_tokens=row[5],
                    cap_usd=self.db.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0])

    def complete(self, messages, *, max_tokens=700, temperature=0, json_mode=False, model=MODEL,
                 request_timeout=120):
        import math
        if isinstance(request_timeout, bool) or not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError('request_timeout must be finite and positive')
        if model not in MODEL_RATES:
            raise ValueError("No verified price/budget bound for model: " + model)
        if not 0 < max_tokens <= 4096:
            raise ValueError("Experiment output cap must be between 1 and 4096 tokens")
        body = {"model": model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "num_retries": 0}
        if model == "gpt-5.6-sol":
            body["reasoning_effort"] = "none"
            body.pop("temperature")  # this Bedrock route rejects temperature even with no reasoning
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        raw = json.dumps(body, ensure_ascii=False).encode()
        if model == "gpt-5.6-sol" and len(raw) + 8192 > 272_000:
            raise ValueError("GPT judge request exceeds the verified short-context price bound")
        # UTF-8 bytes upper-bound text token counts; ample margin covers chat framing.
        # Preserve the original $1/M floor, but use higher rates for expensive judges.
        # Keep 4x headroom for proxy/provider retry uncertainty, including lost responses.
        input_rate, output_rate = MODEL_RATES[model]
        reservation = 4 * ((len(raw) + 8192) * max(1, input_rate)
                           + max_tokens * max(1, output_rate)) / 1_000_000
        self.db.execute("BEGIN IMMEDIATE")
        try:
            used = accounted_exposure(self.db)
            cap = self.db.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0]
            if used + reservation > cap:
                raise BudgetExceeded(f"No dispatch: {used:.4f} + {reservation:.4f} exceeds ${cap:.2f}")
            request_id = self.db.execute(
                "INSERT INTO calls(label,created,reserved,state) VALUES(?,?,?,'pending')",
                (self.label, datetime.now(timezone.utc).isoformat(), reservation)).lastrowid
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        t0 = time.perf_counter()
        artifact = {"request": body, "reservation_usd": reservation, "label": self.label,
                    "rates_per_million": MODEL_RATES[model], "request_timeout_seconds": request_timeout}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("DEBATE_LLM_API_KEY")
        if key:
            headers["Authorization"] = "Bearer " + key
        input_tokens = output_tokens = estimate = None
        try:
            with urlopen(Request(self.base_url + "/chat/completions", data=raw, headers=headers), timeout=request_timeout) as r:
                result = json.load(r)
            artifact["response"] = result
            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            estimate = None
            if (type(input_tokens) is int and type(output_tokens) is int
                    and input_tokens >= 0 and output_tokens >= 0):
                input_rate, output_rate = MODEL_RATES[model]
                estimate = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
            content = result["choices"][0]["message"].get("content")
            if result["choices"][0].get("finish_reason") == "length":
                artifact["truncated"] = True
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty completion")
            self.db.execute("""UPDATE calls SET state=?, input_tokens=?, output_tokens=?,
                estimated_usd=?, seconds=? WHERE id=?""",
                ("ok" if estimate is not None else "usage_missing", input_tokens, output_tokens,
                 estimate, time.perf_counter()-t0, request_id))
            self.db.commit()
            return content
        except BaseException as exc:
            artifact["error"] = type(exc).__name__ + ": " + str(exc)
            self.db.execute("""UPDATE calls SET state='error', input_tokens=?, output_tokens=?,
                estimated_usd=?, seconds=? WHERE id=?""",
                            (input_tokens, output_tokens, estimate, time.perf_counter()-t0, request_id))
            self.db.commit()
            raise
        finally:
            artifact["seconds"] = time.perf_counter()-t0
            path = self.directory / f"call_{request_id:06}.json"
            path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
            if "error" not in artifact and estimate is not None:
                reconcile_success(self.db, request_id, path)

    def text(self, prompt, max_tokens=700):
        return self.complete([{"role": "user", "content": prompt}], max_tokens=max_tokens)
