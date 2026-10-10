"""Append-only reconciliation of successful experiment requests.

Original reservations and call rows remain intact. Successful text responses with
verified usage may replace their *active* reservation with 4x reported cost.
Completed audio bundles retain their recorded per-request 4x bounds; unused
bundle capacity can be released after verification. Failures and incomplete
records keep their entire reservation.
This is conservative accrual, not a provider invoice or a guaranteed billing bound.
"""
from datetime import datetime, timezone
import hashlib
import json
import math

MODEL_RATES = {"google.gemma-4-26b-a4b": (0.13, 0.40),
               "nvidia.nemotron-super-3-120b": (0.15, 0.65),
               "gpt-5.6-sol": (4.40, 22.00)}
SETTLEMENT_MARGIN = 4


class AuditMismatch(ValueError):
    pass


def initialize_accounting(db):
    db.execute("""CREATE TABLE IF NOT EXISTS budget_settlements(
        call_id INTEGER PRIMARY KEY REFERENCES calls(id),
        charge_usd REAL NOT NULL CHECK(charge_usd > 0),
        original_reserved REAL NOT NULL,
        artifact_sha256 TEXT NOT NULL, basis TEXT NOT NULL, created TEXT NOT NULL)""")
    for action in ('UPDATE', 'DELETE'):
        db.execute(f"""CREATE TRIGGER IF NOT EXISTS settlements_no_{action.lower()}
            BEFORE {action} ON budget_settlements
            BEGIN SELECT RAISE(ABORT, 'Budget settlements are append-only'); END""")
    db.commit()


def exposure_query(where=''):
    # A later failed state or changed original reservation cannot retain an old credit.
    return """SELECT coalesce(sum(CASE WHEN c.state='ok' AND c.reserved=s.original_reserved
        THEN coalesce(s.charge_usd,c.reserved) ELSE c.reserved END),0)
        FROM calls c LEFT JOIN budget_settlements s ON c.id=s.call_id""" + where


def accounted_exposure(db, label=None):
    where, args = (' WHERE c.label=?', (label,)) if label is not None else ('', ())
    return db.execute(exposure_query(where), args).fetchone()[0]


def reported_cost(artifact):
    """Recompute text usage, rejecting unknown prices and malformed counters."""
    request = artifact['request']
    model = request['model']
    if model not in MODEL_RATES:
        raise AuditMismatch('No verified text price')
    if 'rates_per_million' in artifact and tuple(artifact['rates_per_million']) != MODEL_RATES[model]:
        raise AuditMismatch('Stored price differs from verified price')
    response = artifact.get('response')
    if response is None:
        return None
    if response.get('model', model) != model:
        raise AuditMismatch('Response model differs from request')
    usage = response.get('usage', {})
    i, o = usage.get('prompt_tokens'), usage.get('completion_tokens')
    if i is None or o is None:
        return None
    if type(i) is not int or type(o) is not int or min(i, o) < 0:
        raise AuditMismatch('Invalid token counters')
    if 'total_tokens' in usage and usage['total_tokens'] != i + o:
        raise AuditMismatch('Token counters do not sum')
    if model == 'gpt-5.6-sol' and i > 272_000:
        raise AuditMismatch('Observed usage exceeds verified pricing tier')
    price_in, price_out = MODEL_RATES[model]
    return i, o, (i*price_in + o*price_out)/1_000_000


def verify_record(row, artifact):
    if artifact['label'] != row['label'] or not math.isclose(
            artifact['reservation_usd'], row['reserved'], rel_tol=0, abs_tol=1e-12):
        raise AuditMismatch('Request identity/reservation mismatch')
    usage = reported_cost(artifact)
    if usage is None:
        if any(row[k] is not None for k in ('input_tokens', 'output_tokens', 'estimated_usd')):
            raise AuditMismatch('Ledger usage has no supporting response')
    else:
        i, o, cost = usage
        if (row['input_tokens'] != i or row['output_tokens'] != o
                or row['estimated_usd'] is None
                or not math.isclose(row['estimated_usd'], cost, rel_tol=0, abs_tol=1e-10)):
            raise AuditMismatch('Ledger usage differs from original response')
    return usage


def successful_charge(row, artifact):
    usage = verify_record(row, artifact)
    if row['state'] != 'ok' or 'error' in artifact or usage is None or usage[1] <= 0:
        return None
    content = artifact.get('response', {}).get('choices', [{}])[0].get('message', {}).get('content')
    if not isinstance(content, str) or not content.strip():
        return None
    return usage[2]*SETTLEMENT_MARGIN


def successful_audio_charge(row, artifact):
    """Release unused capacity only after all recorded audio calls have settled.

    Keep each request's original 4x bound, including UTF-8 or ASR headroom.
    A tiny positive floor preserves the existing append-only table constraint
    for completed bundles that dispatched no requests.
    """
    if artifact['label'] != row['label'] or artifact['reservation_usd'] != row['reserved']:
        raise AuditMismatch('Audio identity/reservation differs')
    if artifact['request']['model'] != 'bounded-audio-bundle':
        raise AuditMismatch('Expected an audio bundle')
    if row['state'] != 'ok':
        return None
    calls = artifact['external_calls']
    limit = artifact.get('max_requests', 24)
    if (type(limit) is not int or not 1 <= limit <= 128
            or not isinstance(calls, list) or len(calls) > limit):
        raise AuditMismatch('Invalid audio request list')
    if any(c.get('state') != 'ok' for c in calls):
        return None
    estimate = bound = 0.
    for call in calls:
        if call['model'] == 'tts-1':
            size = call['characters']
            if type(size) is not int or not 0 < size <= 4096:
                raise AuditMismatch('Invalid TTS character count')
            cost = size * 15 / 1e6
        elif call['model'] == 'whisper-1':
            seconds = call['seconds']
            if not isinstance(seconds, (int, float)) or not 0 < seconds <= 120:
                raise AuditMismatch('Invalid ASR duration')
            cost = math.ceil(seconds) / 60 * .006
        else:
            raise AuditMismatch('Unknown audio price')
        reserve = call['reserved_usd']
        if (not math.isfinite(reserve) or reserve + 1e-12 < 4 * cost
                or not math.isclose(call['estimated_usd'], cost, abs_tol=1e-12)
                or call.get('response_bytes', 0) <= 0):
            raise AuditMismatch('Invalid audio receipt or bound')
        estimate += cost
        bound += reserve
    if (row['estimated_usd'] is None
            or not math.isclose(estimate, row['estimated_usd'], abs_tol=1e-10)
            or bound > row['reserved'] + 1e-10
            or 'seconds' not in artifact
            or not math.isclose(artifact['seconds'], row['seconds'], abs_tol=1e-10)):
        raise AuditMismatch('Audio bundle does not match its completed ledger record')
    return min(row['reserved'], max(1e-9, bound))


def reconcile_success(db, request_id, artifact_path):
    """Append a verified settlement atomically; repeat calls never release twice."""
    raw = artifact_path.read_bytes()
    artifact = json.loads(raw)
    is_audio = artifact['request']['model'] == 'bounded-audio-bundle'
    digest = hashlib.sha256(raw).hexdigest()
    db.execute('BEGIN IMMEDIATE')
    try:
        cursor = db.execute('SELECT * FROM calls WHERE id=?', (request_id,))
        record = cursor.fetchone()
        if record is None:
            raise AuditMismatch('No call record for artifact')
        row = dict(zip((c[0] for c in cursor.description), record))
        charge = (successful_audio_charge if is_audio else successful_charge)(row, artifact)
        if charge is None:
            db.commit()
            return False
        previous = db.execute('SELECT charge_usd,original_reserved,artifact_sha256 FROM budget_settlements WHERE call_id=?',
                              (request_id,)).fetchone()
        if previous is not None:
            if (previous[0], previous[1], previous[2]) != (charge, row['reserved'], digest):
                raise AuditMismatch('Existing settlement evidence changed')
            db.commit()
            return False
        db.execute('INSERT INTO budget_settlements VALUES(?,?,?,?,?,?)',
                   (request_id, charge, row['reserved'], digest,
                    ('Verified completed audio: retain original per-request 4x bounds; release unused bundle capacity'
                     if is_audio else '4x verified reported text cost; original reservation retained for audit'),
                    datetime.now(timezone.utc).isoformat()))
        db.commit()
        return True
    except BaseException:
        db.rollback()
        raise
