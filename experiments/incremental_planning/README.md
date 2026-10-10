# Motion experiments and offline audits

The fixed experimental reference is recorded in
[CURRENT_BASELINE.json](CURRENT_BASELINE.json). Its archived source and settings
are immutable. Current runtime documentation is in
[the streaming guide](../../src/streaming/README.md).

## Maintained support

- `benchmark_listening_motion_live.py` implements the server-paced six-turn
  harness. `run_streamed_motion_v69.py` and `run_streamed_motion_v70.py` record
  the two completed configurations; their existing run IDs are single-use.
- `audio_probe.py`, `reconcile_budget.py` and the shared `streaming` accounting
  modules retain pre-dispatch reservations, cumulative caps and usage evidence.
- `audit_handoff_v70.py`, `verify_audio_v70.py`, `audit_body_preparation.py` and
  `report_streamed_motion_v70.py` inspect saved artifacts without provider calls.
- Older benchmark helpers retained here are dependencies of offline regression
  tests or historical report readers. Their numbered configurations are not the
  current baseline and their saved approvals do not authorize new runs.

The runners still require the configured local gateway, external model assets,
prepared source pools and credentials. Do not reset `run/cost.sqlite`, overwrite
an existing run or recycle an old approval to reproduce a result. A fresh paid
experiment needs a distinct run ID, current authorization and the existing
cumulative budget guard.

## Historical tools and data

Unreferenced one-off probes, retries and intermediate version launchers were
removed from this directory and stored in
[history/20261010-tools.tar.gz](history/20261010-tools.tar.gz). The accompanying
[manifest](history/20261010-tools.json) records original paths and SHA-256 hashes
for all 63 scripts. These are reference sources: restore the original paths and
matching historical code before attempting reproduction. They are not supported
launchers for the cleaned runtime.

Expanded copies remain locally in `local_history/`; generated request logs,
SQLite ledgers, speech artifacts and copied source trees remain local and are
ignored by Git. Small historical inputs required by tests are explicitly stored
under `tests/fixtures/experiments/`, rather than discovered in live run folders.
The retained v53 manifest is a historical input for the native-revision test.

The baseline archive and historical tool archive contain no API credential file.
To verify the baseline independently of current source:

```bash
python experiments/incremental_planning/baselines/v70-20261010/verify.py
```
