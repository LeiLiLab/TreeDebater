"""Audit every request offline; optionally append verified successful settlements.

No network/model calls. --apply backs up SQLite first, preserves all calls and
artifacts, and never changes the approved cap. Pending work blocks bulk apply.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from streaming.experiment_accounting import (MODEL_RATES, SETTLEMENT_MARGIN, accounted_exposure,
                                             initialize_accounting, reconcile_success,
                                             successful_charge, successful_audio_charge, verify_record)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def audit(directory, apply=False):
    db = sqlite3.connect(f'file:{directory / "cost.sqlite"}?mode={"rw" if apply else "ro"}', uri=True)
    db.row_factory = sqlite3.Row
    cap = db.execute('select cap from budget where id=1').fetchone()[0]
    rows = [dict(r) for r in db.execute('select * from calls order by id')]
    original_rows_hash = fingerprint(rows)
    models = defaultdict(lambda: dict(entries=0, input_tokens=0, output_tokens=0,
                                     usage_estimate_usd=0., original_reservations_usd=0.))
    candidate_exposure = 0.
    artifacts, candidates, retained, issues, response_ids = {}, [], [], [], set()
    audio_calls = audio_candidates = 0
    for row in rows:
        try:
            path = directory/f"call_{row['id']:06}.json"
            raw = path.read_bytes();artifacts[str(row['id'])] = hashlib.sha256(raw).hexdigest()
            a = json.loads(raw);model = a['request']['model'];m = models[model];m['entries'] += 1
            m['original_reservations_usd'] += row['reserved']
            if a['label'] != row['label'] or a['reservation_usd'] != row['reserved']:
                raise ValueError('Identity or reservation differs')
            response_id = a.get('response', {}).get('id')
            if response_id:
                if response_id in response_ids:raise ValueError('Repeated provider response ID')
                response_ids.add(response_id)
            if model == 'bounded-audio-bundle':
                calls = a['external_calls'];audio_calls += len(calls)
                if sum(c['reserved_usd'] for c in calls) > row['reserved'] + 1e-10:
                    raise ValueError('Audio internal reservations exceed bundle')
                estimate = 0.
                for c in calls:
                    if c['model'] == 'tts-1':cost = c['characters']*15/1e6
                    elif c['model'] == 'whisper-1':cost = math.ceil(c['seconds'])/60*.006
                    else:raise ValueError('Unknown audio model')
                    if not math.isclose(cost, c['estimated_usd'], abs_tol=1e-10):
                        raise ValueError('Audio price mismatch')
                    if c['state'] == 'ok':estimate += cost
                if not math.isclose(estimate, row['estimated_usd'], abs_tol=1e-10):
                    raise ValueError('Audio ledger estimate mismatch')
                m['usage_estimate_usd'] += estimate
                charge = successful_audio_charge(row, a)
                if charge is not None:
                    audio_candidates += 1
            else:
                usage = verify_record(row, a)
                if usage:
                    m['input_tokens'] += usage[0];m['output_tokens'] += usage[1];m['usage_estimate_usd'] += usage[2]
                charge = successful_charge(row, a)
            if charge is None:
                retained.append(dict(id=row['id'], label=row['label'], state=row['state'],
                                     reservation_usd=row['reserved'], known_usage_estimate_usd=row['estimated_usd']))
                candidate_exposure += row['reserved']
            else:
                candidates.append(row['id']);candidate_exposure += charge
        except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
            issues.append(dict(id=row['id'], error=str(exc)))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), cap_usd=cap,
                  entries=len(rows), original_calls_sha256=original_rows_hash,
                  artifacts_manifest_sha256=fingerprint(artifacts), artifacts_checked=len(artifacts),
                  model_totals=dict(models), original_reservations_usd=sum(r['reserved'] for r in rows),
                  usage_estimate_usd=sum(m['usage_estimate_usd'] for m in models.values()),
                  candidate_successful_text_settlements=len(candidates)-audio_candidates,
                  candidate_successful_audio_settlements=audio_candidates, retained_full_reservations=retained,
                  settlement_margin=SETTLEMENT_MARGIN, proposed_accounted_exposure_usd=candidate_exposure,
                  proposed_available_budget_usd=cap-candidate_exposure, external_audio_http_calls=audio_calls,
                  unknown_usage_request_ids=[r['id'] for r in rows if r['estimated_usd'] is None],
                  pending=sum(r['state']=='pending' for r in rows), issues=issues, applied=False,
                  verified_text_rates_per_million=MODEL_RATES,
                  pricing_sources=['https://aws.amazon.com/bedrock/pricing/',
                                   'https://docs.aws.amazon.com/en_en/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html',
                                   'https://developers.openai.com/api/docs/models/tts-1',
                                   'https://developers.openai.com/api/docs/models/whisper-1'],
                  policy='4x verified text cost; completed audio retains recorded per-request4x bounds. Failed/unknown requests retain full original reservations. Originals are never rewritten.',
                  limitations=['Usage estimates are not settled invoices; no provider billing statement was available.',
                               'Request logs cannot independently rule out unreported upstream retries; 4x accrual is a conservative allowance, not a proof of a billing upper bound.'])
    if apply:
        if issues or report['pending']:
            raise ValueError('Audit issues or pending work prevent bulk reconciliation: '+json.dumps(issues))
        backup_path = directory/('cost-before-reconciliation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.sqlite')
        if not backup_path.exists():
            backup = sqlite3.connect(backup_path);db.backup(backup);backup.close()
        initialize_accounting(db)
        report['new_settlements'] = sum(reconcile_success(db, i, directory/f'call_{i:06}.json') for i in candidates)
        after_rows = [dict(r) for r in db.execute('select * from calls order by id')]
        if fingerprint(after_rows) != original_rows_hash:
            raise ValueError('Original call rows changed during audit')
        if any(hashlib.sha256((directory/f'call_{int(i):06}.json').read_bytes()).hexdigest()!=digest
               for i,digest in artifacts.items()):
            raise ValueError('Original artifacts changed during audit')
        if db.execute('select cap from budget where id=1').fetchone()[0] != cap:
            raise ValueError('Cap changed during audit')
        report.update(applied=True, original_calls_unchanged=True, original_artifacts_unchanged=True,
                      actual_accounted_exposure_usd=accounted_exposure(db))
        if not math.isclose(report['actual_accounted_exposure_usd'], candidate_exposure, abs_tol=1e-8):
            raise ValueError('Applied exposure differs from audited exposure')
    return report


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    report=audit(ROOT/'experiments/incremental_planning/run', args.apply)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('entries','usage_estimate_usd','original_reservations_usd',
                                          'proposed_accounted_exposure_usd','proposed_available_budget_usd','issues','applied')},indent=2))
