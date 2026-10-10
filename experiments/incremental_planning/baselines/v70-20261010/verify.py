"""Offline integrity check; --workspace also detects changes from the baseline."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace', type=Path)
args = parser.parse_args()
base = Path(__file__).resolve().parent
files = json.loads((base / 'files.json').read_text())
failures = []
with tarfile.open(base / 'snapshot.tar.gz', 'r:gz') as archive:
    for relative, expected in files.items():
        member = archive.extractfile('snapshot/' + relative)
        if member is None or hashlib.sha256(member.read()).hexdigest() != expected:
            failures.append('archive:' + relative)
        if args.workspace:
            target = args.workspace / relative
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                failures.append(str(target))
expected = (base / 'snapshot.tar.gz.sha256').read_text().split()[0]
if hashlib.sha256((base / 'snapshot.tar.gz').read_bytes()).hexdigest() != expected:
    failures.append('snapshot.tar.gz')
print(json.dumps({'checked_files': len(files), 'failures': failures}, indent=2))
raise SystemExit(bool(failures))
