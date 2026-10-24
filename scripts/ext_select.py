# D-053 ext-train: pick the public SWE-style tasks (SWE-rebench, SWE-bench-extra, SWE-rebench-V2, SWE-Gym) that
# can serve as training tasks in OUR format and grader, in verification priority order.
# Filters: Python >= 3.8 where the source states it (older environments rarely install on the scorer's 3.13;
# verification decides the rest, newest first), created 2021+, pub129-shaped gold (<= 4 files, <= 150 changed lines, no binary/rename), non-empty
# FAIL_TO_PASS, issue text rated well specified (SWE-rebench llm_score.issue_text_score <= 1 when present).
# Exclusions (repo level): pub129 repos, every priv60 / priv60r2 / mine3 repo, the SWE-bench test-split repos and
# the ext40 / gym40 / lite40 eval repos. Dedupe across sources by (repo, base_commit, gold patch).
# Ids follow pub129 (`<repo short>_<number>`, owner-prefixed on collision); the source id goes to meta.
# Writes <out>/candidates.jsonl (source rows + source/orig_id/priority) and a summary.
# Usage: uv run --with pyarrow python scripts/mining/ext_select.py [--out data/mine3/ext]
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

SOURCES = {  # source dir -> parquet glob
    'swe_rebench': 'data/test-*.parquet',
    'swebench_extra': 'data/train-*.parquet',
    'swe_rebench_v2': 'data/train-*.parquet',
    'swe_gym': 'train.parquet',
}
PUBLIC = {'fastapi/fastapi', 'Textualize/rich', 'psf/requests', 'encode/httpx'}


def side_meta(src: str) -> dict[str, dict]:
    """install python version, llm issue score, license, from the source parquet (not kept by `swe ext import`)."""
    out = {}
    for f in sorted(Path('data/ext', src).glob(SOURCES[src])):
        cols = [c for c in ('instance_id', 'install_config', 'meta', 'license_name', 'license')
                if c in pq.ParquetFile(f).schema_arrow.names]
        for r in pq.read_table(f, columns=cols).to_pylist():
            ic = r.get('install_config') or {}
            if isinstance(ic, str):
                ic = json.loads(ic or '{}')
            meta = r.get('meta') or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}
            score = (meta.get('llm_score') or {}) if isinstance(meta, dict) else {}
            py = str(ic.get('python') or '')
            m = re.match(r'python_base_(\d)(\d+)', str(ic.get('base_image_name') or ''))
            if not py and m:  # SWE-rebench-V2: python_base_310 -> 3.10
                py = f'{m.group(1)}.{m.group(2)}'
            out[r['instance_id']] = {'python': py,
                                     'issue_score': score.get('issue_text_score'),
                                     'license': r.get('license_name') or r.get('license')}
    return out


def pyver(v: str) -> tuple[int, int]:
    m = re.match(r'(\d+)\.(\d+)', v or '')
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def gold_shape(patch: str) -> tuple[int, int, bool]:
    files = re.findall(r'^\+\+\+ b/(\S+)', patch, re.MULTILINE)
    lines = sum(1 for x in patch.splitlines() if x[:1] in '+-' and x[:3] not in ('+++', '---'))
    bad = 'GIT binary patch' in patch or '\nrename from ' in patch
    return len(files), lines, bad


def short_id(repo: str, orig: str) -> str:
    num = re.search(r'(\d+)$', orig)
    return repo.rsplit('/', 1)[-1].lower().replace('-', '_').replace('.', '_') + f'_{num.group(1) if num else "0"}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('data/mine3/ext'))
    ap.add_argument('--since', default='2021-01-01')
    ap.add_argument('--min-python', default='3.8')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    excluded = set(PUBLIC)
    for p in ('data/priv60/tasks.jsonl', 'data/priv60r2/tasks.jsonl', 'data/mine3/eval/tasks.jsonl',
              'data/mine3/train/tasks.jsonl', 'data/ext/swebench_verified/tasks.jsonl',
              'data/ext/swebench_lite/tasks.jsonl', 'data/ext/swe_gym_lite/tasks.jsonl'):
        if Path(p).exists():
            excluded |= {json.loads(line)['repo'] for line in open(p, encoding='utf-8')}
    excluded |= set(pq.read_table('data/ext/swebench/test.parquet', columns=['repo']).column('repo').to_pylist())
    for p in ('data/mine3/mining/repos_eval.txt',):
        if Path(p).exists():
            excluded |= set(Path(p).read_text().split())

    minv = pyver(a.min_python)
    why, rows, seen, ids = Counter(), [], set(), set()
    per_repo = defaultdict(int)
    for src in SOURCES:
        tp = Path('data/ext', src, 'tasks.jsonl')
        if not tp.exists():
            print('missing', tp)
            continue
        meta = side_meta(src) if src != 'swe_gym' else {}
        for line in tp.open(encoding='utf-8'):
            t = json.loads(line)
            m = meta.get(t['instance_id'], {})
            if t['repo'] in excluded:
                why['excluded repo'] += 1
                continue
            if (t.get('created_at') or '')[:10] < a.since:
                why['older than since'] += 1
                continue
            if m.get('python') and pyver(m['python']) < minv:
                why['python < min'] += 1
                continue
            if m.get('issue_score') is not None and m['issue_score'] > 1:
                why['issue underspecified'] += 1
                continue
            if not t.get('FAIL_TO_PASS') or not t.get('test_patch'):
                why['no F2P / test patch'] += 1
                continue
            nf, nl, bad = gold_shape(t['patch'])
            if bad or nf == 0 or nf > 4 or nl > 150:
                why['gold out of band'] += 1
                continue
            key = hashlib.sha256(f"{t['repo']}|{t['base_commit']}|{t['patch']}".encode()).hexdigest()
            if key in seen:
                why['duplicate'] += 1
                continue
            seen.add(key)
            iid = short_id(t['repo'], t['instance_id'])
            if iid in ids:
                iid = t['repo'].replace('/', '__').lower().replace('-', '_').replace('.', '_') + '_' + iid.rsplit('_', 1)[1]
            ids.add(iid)
            per_repo[t['repo']] += 1
            # priority: newer, smaller, closer to pub129 (1 file, ~12 lines), fewer tasks per repo first
            pri = (-int((t.get('created_at') or '0000')[:4]), abs(nf - 1), abs(nl - 12) // 10, per_repo[t['repo']])
            rows.append({**t, 'instance_id': iid, 'orig_id': t['instance_id'], 'source': src,
                         'license': m.get('license'), 'python': m.get('python'), '_pri': pri})
            why['kept'] += 1
    rows.sort(key=lambda r: r['_pri'])
    with (a.out / 'candidates.jsonl').open('w', encoding='utf-8') as f:
        for i, r in enumerate(rows):
            r.pop('_pri')
            r['priority'] = i
            f.write(json.dumps(r) + '\n')
    src = Counter(r['source'] for r in rows)
    print(dict(why))
    print(f'candidates {len(rows)} from {len({r["repo"] for r in rows})} repos, by source {dict(src)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
