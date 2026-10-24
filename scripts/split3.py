# D-053: split the round-3 PR details by REPO into priv-eval and priv-train inputs for `repograph mine`.
# priv-eval repos = every priv60 / priv60r2 repo + a stable-hash 12 % of the new repos that have a PR merged
# 2025-07+ (after the model's data); eval keeps <= 3 PRs per repo merged 2025-07+, smallest first (pub129 is
# small: median 12 changed lines, 1 file). priv-train = all other repos, PRs merged since --since, <= 8 per repo.
# PRs already in priv60 / priv60r2 are dropped. Writes <out>/details_eval.jsonl, details_train.jsonl,
# repos_eval.txt, repos_train.txt.
# Usage: uv run python scripts/mining/split3.py [--mining data/mine3/mining] [--out data/mine3/mining]
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

EVAL_SINCE = '2025-07-01'


def short_id(repo: str, number: int) -> str:
    return repo.rsplit('/', 1)[-1].lower().replace('-', '_').replace('.', '_') + f'_{number}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mining', type=Path, default=Path('data/mine3/mining'))
    ap.add_argument('--out', type=Path, default=Path('data/mine3/mining'))
    ap.add_argument('--since', default='2024-01-01')
    ap.add_argument('--eval-share', type=float, default=0.12)
    ap.add_argument('--eval-per-repo', type=int, default=3)
    ap.add_argument('--train-per-repo', type=int, default=8)
    a = ap.parse_args()

    old_ids, old_repos = set(), set()
    for p in ('data/priv60/tasks.jsonl', 'data/priv60r2/tasks.jsonl'):
        for line in Path(p).open(encoding='utf-8'):
            t = json.loads(line)
            old_ids.add(t['instance_id'])
            old_repos.add(t['repo'])
    size = {}
    for line in (a.mining / 'prs.jsonl').open(encoding='utf-8'):
        x = json.loads(line)
        for p in x.get('prs', []):
            size[(x['repo'], p['number'])] = p['src_add'] + p['src_del']
    by_repo = defaultdict(list)
    for line in (a.mining / 'pr_details.jsonl').open(encoding='utf-8'):
        d = json.loads(line)
        if d.get('error') or not d.get('merge_commit') or not d.get('issue') or not d.get('parents'):
            continue
        if short_id(d['repo'], d['number']) in old_ids:
            continue
        d['_size'] = size.get((d['repo'], d['number']), 10 ** 6)
        by_repo[d['repo']].append(d)

    def h(repo: str) -> float:
        return int(hashlib.sha256(f'd053:{repo}'.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    eval_rows, train_rows, eval_repos, train_repos = [], [], [], []
    for repo, prs in sorted(by_repo.items()):
        recent = sorted([p for p in prs if (p.get('merged_at') or '') >= EVAL_SINCE], key=lambda p: p['_size'])
        if repo in old_repos or (recent and h(repo) < a.eval_share):
            if recent:
                eval_repos.append(repo)
                eval_rows += recent[:a.eval_per_repo]
            continue
        keep = sorted([p for p in prs if (p.get('merged_at') or '') >= a.since], key=lambda p: p['_size'])
        if keep:
            train_repos.append(repo)
            train_rows += keep[:a.train_per_repo]
    for name, rows in (('details_eval.jsonl', eval_rows), ('details_train.jsonl', train_rows)):
        with (a.out / name).open('w', encoding='utf-8') as f:
            for r in rows:
                r.pop('_size', None)
                f.write(json.dumps(r) + '\n')
    (a.out / 'repos_eval.txt').write_text('\n'.join(eval_repos) + '\n', encoding='utf-8')
    (a.out / 'repos_train.txt').write_text('\n'.join(train_repos) + '\n', encoding='utf-8')
    assert not set(eval_repos) & set(train_repos)
    print(f'eval: {len(eval_rows)} PRs from {len(eval_repos)} repos ({len(set(eval_repos) & old_repos)} priv60 repos)')
    print(f'train: {len(train_rows)} PRs from {len(train_repos)} repos')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
