"""Optional bounded settings for the shared, publicly accessible research demo."""
import os
from datetime import datetime, timezone

from fastapi import HTTPException

from .schemas import SessionSettings


def public_settings():
    return SessionSettings(
        motion="Learning to be a good writer still matters in the age of AI",
        ai_model="gpt-4o-mini",
        helper_model="gpt-4o-mini",
        streaming={"output": {"refinement_model": "gpt-4o-mini"}},
    )


def validate_public(settings):
    defaults = public_settings()
    if (settings.ai_model != defaults.ai_model
            or settings.helper_model not in (None, defaults.helper_model)
            or settings.streaming != defaults.streaming
            or settings.claim_pool_size > 4):
        raise HTTPException(422, "The public demo uses fixed GPT-4o-mini model and streaming settings. Reload the page to restore defaults.")
    if any(settings.budgets[stage] > limit for stage, limit in
           (("opening", 240), ("rebuttal", 240), ("closing", 120))):
        raise HTTPException(422, "Public demo limits: opening/rebuttal 240 seconds, closing 120 seconds.")


def check_quota(storage, session_id):
    """Persist the daily cap; idempotent retries do not consume another slot."""
    day = datetime.now(timezone.utc).date().isoformat()
    cap = int(os.getenv("DEBATE_APP_PUBLIC_DAILY_SESSIONS", "20"))
    with storage.lock, storage.db:
        storage.db.execute("CREATE TABLE IF NOT EXISTS public_usage(day TEXT, session_id TEXT PRIMARY KEY)")
        if storage.db.execute("SELECT 1 FROM public_usage WHERE session_id=?", (session_id,)).fetchone():
            return
        count = storage.db.execute("SELECT COUNT(*) FROM public_usage WHERE day=?", (day,)).fetchone()[0]
        if count >= cap:
            raise HTTPException(429, "Today's public demo allowance is used up. Please return tomorrow (UTC).")


def record_usage(storage, session_id):
    with storage.lock, storage.db:
        storage.db.execute("INSERT OR IGNORE INTO public_usage VALUES(?, ?)",
                           (datetime.now(timezone.utc).date().isoformat(), session_id))
