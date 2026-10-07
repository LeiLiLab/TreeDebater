"""Build or verify persistent hybrid indexes for a directory of prepared pools."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from utils.hybrid_rehearsal import HybridRehearsalRetriever
from utils.local_encoder import LocalEncoder
from utils.rehearsal_index_cache import cache_path


class PreparedPool:
    """Read-only tree view for offline indexing, without API/client dependencies."""
    @classmethod
    def from_json(cls, value):
        tree = cls()
        tree.side = value['side']
        tree.nodes = []
        def visit(raw, parent=None):
            node = SimpleNamespace(claim=raw['claim'], argument=raw.get('argument', []),
                side=raw['side'], status=raw.get('status', 'prepared'),
                position_status=raw.get('position_status', 'current'), parent=parent, children=[])
            tree.nodes.append(node)
            node.children = [visit(child, node) for child in raw.get('children', [])]
            return node
        tree.root = visit(value['structure'])
        return tree

    def get_node_by_side(self, side):
        return [n for n in self.nodes if n.side == side and n.status != 'root']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pool_dir', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    pool_dir = args.pool_dir.resolve()
    cache_dir = (args.cache_dir or pool_dir / 'retrieval_indexes').resolve()
    files = sorted(pool_dir.glob('*_pool_*.json'))
    if not files:
        raise ValueError('No prepared pool files found')
    pools, sources = {}, {}
    for path in files:
        slug, side = path.stem.rsplit('_pool_', 1)
        if side not in ('for', 'against'):
            raise ValueError(f'Unexpected pool side: {path}')
        raw = path.read_bytes()
        trees = [PreparedPool.from_json(row[0]['tree_structure']) for row in json.loads(raw)]
        if any(t.side != side for t in trees):
            raise ValueError(f'Pool side mismatch: {path}')
        pools.setdefault(slug, {})[side] = trees
        sources[path.name] = dict(sha256=hashlib.sha256(raw).hexdigest(), trees=len(trees))
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoder = LocalEncoder(args.model, args.threads)
    if args.verify_only:
        def forbidden(*a, **kw):
            raise AssertionError('Disk cache miss during verification; material encoding is forbidden')
        encoder.encode = forbidden
    report = dict(status='running', pool_dir=str(pool_dir), cache_dir=str(cache_dir), sources=sources,
                  model=args.model, threads=args.threads, new_api_calls=0,
                  missing_pools=[dict(motion=slug, side=side) for slug, sides in pools.items()
                                 for side in ('for', 'against') if side not in sides], indexes=[])
    report_name = 'verification.json' if args.verify_only else 'manifest.json'
    def save():
        path = cache_dir / report_name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)
    start = time.perf_counter()
    try:
        for slug, sides in pools.items():
            for side in ('for', 'against'):
                opposite = 'against' if side == 'for' else 'for'
                for attack in (True, False):
                    index = HybridRehearsalRetriever(encoder, cache_dir=cache_dir)
                    before = time.perf_counter()
                    index.prepare(sides.get(side, []), sides.get(opposite, []), side, opposite, attack)
                    path = cache_path(cache_dir, encoder, index.signature) if index.records else None
                    if index.records:
                        assert path.is_file(), 'Index was not persisted'
                        if args.verify_only:
                            assert index.disk_cache_status == 'hit'
                    report['indexes'].append(dict(motion=slug, side=side, group='attack_rebut' if attack else 'support',
                        records=len(index.records), status=index.disk_cache_status, seconds=time.perf_counter()-before,
                        file=path.name if path else None,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path else None))
                    save()
            print(f'{len(report["indexes"])} / {len(pools)*4} indexes: {slug}', flush=True)
        assert {p.name for p in pool_dir.glob('*_pool_*.json')} == set(sources), 'Pool inventory changed during build'
        for name, source in sources.items():
            assert hashlib.sha256((pool_dir / name).read_bytes()).hexdigest() == source['sha256'], name
        if args.verify_only:
            assert encoder.process is None
            report['encoder_started'] = False
        report.update(status='completed', elapsed_seconds=time.perf_counter()-start,
                      motions=len(pools), index_count=len(report['indexes']),
                      bytes=sum((cache_dir / name).stat().st_size for name in {r['file'] for r in report['indexes'] if r['file']}))
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        encoder.close()
        save()


if __name__ == '__main__':
    main()
