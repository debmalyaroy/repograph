# D-053 round 3 mining (priv-eval expansion + priv-train): GitHub search -> merged PRs that close an issue and
# touch source .py + tests -> PR details, resumable at every stage. Read-only GitHub calls through `gh`.
# Usage: uv run python scripts/mining/mine3.py {search|prs|details} [--root data/mine3]
#   search   repos.jsonl: pure-Python repos, 30..5000 stars, pushed since 2026-01, paginated per query
#   prs      prs.jsonl: per repo the last 100 merged PRs since --since with src .py + tests + a closing issue
#   details  pr_details.jsonl: merge commit, parents, issue text, file list (the input of `repograph mine`)
#   noissue  noissue_details.jsonl (--repos file, training repos only): merged PRs with src .py + tests but NO
#            closing issue; the PR title + description stands in for the issue (`issue.from_pr`), synth-train
import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TEST = re.compile(r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$')
PUBLIC = {'fastapi/fastapi', 'Textualize/rich', 'psf/requests', 'encode/httpx'}
STARS = ['30..80', '80..150', '150..300', '300..600', '600..1200', '1200..2500', '2500..5000']
TOPICS = ['', 'topic:cli', 'topic:library', 'topic:api', 'topic:parser', 'topic:http', 'topic:testing', 'topic:data',
          'topic:asyncio', 'topic:pydantic', 'topic:django', 'topic:flask', 'topic:automation', 'topic:sdk',
          'topic:yaml', 'topic:json', 'topic:validation', 'topic:database', 'topic:devtools', 'topic:linter']
PRS_Q = '''query($o:String!,$n:String!){repository(owner:$o,name:$n){
 isFork isArchived primaryLanguage{name} languages(first:8,orderBy:{field:SIZE,direction:DESC}){totalSize edges{size node{name}}}
 root: object(expression:"HEAD:"){... on Tree{entries{name type}}}
 pullRequests(states:MERGED,last:100,orderBy:{field:UPDATED_AT,direction:ASC}){nodes{
   number title mergedAt baseRefName additions deletions changedFiles
   closingIssuesReferences(first:2){nodes{number title createdAt bodyText}}
   files(first:40){nodes{path additions deletions changeType}}}}}}'''
DETAIL_Q = '''query($o:String!,$n:String!,$num:Int!){repository(owner:$o,name:$n){pullRequest(number:$num){
 number title url createdAt mergedAt baseRefName mergeCommit{oid parents(first:2){nodes{oid}}}
 commits(first:1){totalCount} additions deletions
 closingIssuesReferences(first:1){nodes{number title body createdAt url}}
 files(first:60){nodes{path additions deletions changeType}}}}}'''
NOISSUE_Q = '''query($o:String!,$n:String!){repository(owner:$o,name:$n){
 pullRequests(states:MERGED,last:100,orderBy:{field:UPDATED_AT,direction:ASC}){nodes{
   number title url bodyText createdAt mergedAt mergeCommit{oid parents(first:2){nodes{oid}}}
   closingIssuesReferences(first:1){totalCount}
   files(first:40){nodes{path additions deletions}}}}}}'''


def gh(args: list[str], tries: int = 4) -> dict | None:
    for attempt in range(tries):
        r = subprocess.run(['gh', 'api', *args], capture_output=True, text=True, encoding='utf-8')
        if r.returncode == 0:
            return json.loads(r.stdout)
        wait = 60 if 'rate limit' in r.stderr.lower() else 5 * (attempt + 1)
        print(f'  gh retry {attempt + 1} in {wait}s: {r.stderr.strip()[:120]}', flush=True)
        time.sleep(wait)
    return None


def done_keys(path: Path, key) -> set:
    if not path.exists():
        return set()
    return {key(json.loads(line)) for line in path.open(encoding='utf-8') if line.strip()}


def search(root: Path) -> None:
    out = root / 'repos.jsonl'
    seen = done_keys(out, lambda d: d['full_name'])
    queries: set[str] = set()
    qdone = root / 'search_done.txt'
    if qdone.exists():
        queries = set(qdone.read_text(encoding='utf-8').splitlines())
    with out.open('a', encoding='utf-8') as f, qdone.open('a', encoding='utf-8') as qf:
        for stars in STARS:
            for topic in TOPICS:
                q = f'language:python stars:{stars} pushed:>2026-01-01 archived:false fork:false {topic}'.strip()
                for page in (1, 2, 3):
                    tag = f'{q}|{page}'
                    if tag in queries:
                        continue
                    d = gh(['-X', 'GET', 'search/repositories', '-f', f'q={q}', '-f', 'sort=updated',
                            '-f', 'per_page=100', '-f', f'page={page}'])
                    time.sleep(2.2)  # search API: 30 requests per minute
                    items = (d or {}).get('items') or []
                    new = 0
                    for it in items:
                        name = it['full_name']
                        if name in seen or name in PUBLIC:
                            continue
                        seen.add(name)
                        new += 1
                        f.write(json.dumps({'full_name': name, 'stars': it['stargazers_count'], 'size': it['size'],
                                            'created_at': it['created_at'], 'pushed_at': it['pushed_at'],
                                            'topics': it.get('topics', []), 'license': (it.get('license') or {}).get('spdx_id'),
                                            'description': it.get('description')}) + '\n')
                    f.flush()
                    qf.write(tag + '\n')
                    qf.flush()
                    print(f'{stars:>10} {topic or "-":18} p{page}: {len(items)} items, {new} new, total {len(seen)}', flush=True)
                    if len(items) < 100:
                        break


def prs_one(full: str, since: str) -> dict:
    o, n = full.split('/')
    d = gh(['graphql', '-f', f'query={PRS_Q}', '-f', f'o={o}', '-f', f'n={n}'])
    repo = ((d or {}).get('data') or {}).get('repository')
    if not repo:
        return {'repo': full, 'error': 'none'}
    tot = repo['languages']['totalSize'] or 1
    langs = {e['node']['name']: e['size'] / tot for e in repo['languages']['edges']}
    native = sum(langs.get(k, 0) for k in ('C', 'C++', 'Rust', 'Cython', 'Go', 'Fortran', 'CUDA'))
    root = {e['name'] for e in ((repo.get('root') or {}).get('entries') or [])}
    prs = []
    for p in repo['pullRequests']['nodes']:
        if not p['mergedAt'] or p['mergedAt'] < since:
            continue
        files = (p.get('files') or {}).get('nodes') or []
        src = [f for f in files if f['path'].endswith('.py') and not TEST.search(f['path'])]
        tst = [f for f in files if TEST.search(f['path']) and f['path'].endswith('.py')]
        issues = (p.get('closingIssuesReferences') or {}).get('nodes') or []
        if not (src and tst and issues):
            continue
        prs.append({'number': p['number'], 'title': p['title'], 'merged': p['mergedAt'], 'base': p['baseRefName'],
                    'src_files': len(src), 'test_files': len(tst), 'other_files': len(files) - len(src) - len(tst),
                    'src_add': sum(f['additions'] for f in src), 'src_del': sum(f['deletions'] for f in src),
                    'issue': issues[0]['number'], 'issue_title': issues[0]['title'],
                    'issue_chars': len(issues[0]['bodyText'] or '') + len(issues[0]['title'])})
    return {'repo': full, 'py': round(langs.get('Python', 0), 3), 'native': round(native, 3),
            'archived': repo.get('isArchived'), 'pyproject': bool(root & {'pyproject.toml', 'setup.py', 'setup.cfg'}),
            'tests_dir': bool(root & {'tests', 'test'}), 'root': sorted(root)[:40], 'prs': prs}


def prs(root: Path, since: str, workers: int) -> None:
    repos = [json.loads(line)['full_name'] for line in (root / 'repos.jsonl').open(encoding='utf-8')]
    out = root / 'prs.jsonl'
    done = done_keys(out, lambda d: d['repo'])
    todo = [r for r in repos if r not in done]
    print('repos', len(repos), 'todo', len(todo), flush=True)
    with out.open('a', encoding='utf-8') as f, ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(lambda r: prs_one(r, since), todo)):
            f.write(json.dumps(res) + '\n')
            f.flush()
            if i % 50 == 0:
                print(i, res['repo'], len(res.get('prs', [])), flush=True)


def eligible(x: dict) -> bool:
    return 'prs' in x and x['py'] >= 0.85 and x['native'] < 0.05 and x['pyproject'] and not x.get('archived')


def detail_one(job: tuple[str, int]) -> dict:
    repo, num = job
    o, n = repo.split('/')
    d = gh(['graphql', '-f', f'query={DETAIL_Q}', '-f', f'o={o}', '-f', f'n={n}', '-F', f'num={num}'])
    pr = ((((d or {}).get('data') or {}).get('repository')) or {}).get('pullRequest')
    if not pr:
        return {'repo': repo, 'number': num, 'error': 'none'}
    files = (pr.get('files') or {}).get('nodes') or []
    iss = ((pr.get('closingIssuesReferences') or {}).get('nodes') or [None])[0]
    mc = pr.get('mergeCommit') or {}
    return {'repo': repo, 'number': num, 'title': pr['title'], 'url': pr['url'], 'created_at': pr['createdAt'],
            'merged_at': pr['mergedAt'], 'merge_commit': mc.get('oid'),
            'parents': [p['oid'] for p in ((mc.get('parents') or {}).get('nodes') or [])],
            'n_commits': (pr.get('commits') or {}).get('totalCount'), 'issue': iss,
            'src_files': [f['path'] for f in files if f['path'].endswith('.py') and not TEST.search(f['path'])],
            'test_files': [f['path'] for f in files if TEST.search(f['path'])],
            'other_files': [f['path'] for f in files if not f['path'].endswith('.py') and not TEST.search(f['path'])]}


def details(root: Path, workers: int) -> None:
    jobs = []
    for line in (root / 'prs.jsonl').open(encoding='utf-8'):
        x = json.loads(line)
        if eligible(x):
            # pub129 band: median 12 changed lines, p90 4 files; keep up to 4 src files, 150 added src lines
            jobs += [(x['repo'], p['number']) for p in x['prs'] if p['src_files'] <= 4 and p['src_add'] <= 150]
    out = root / 'pr_details.jsonl'
    done = done_keys(out, lambda d: (d['repo'], d['number']))
    todo = [j for j in jobs if j not in done]
    print('jobs', len(jobs), 'todo', len(todo), flush=True)
    with out.open('a', encoding='utf-8') as f, ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(detail_one, todo)):
            f.write(json.dumps(res) + '\n')
            f.flush()
            if i % 100 == 0:
                print(i, res['repo'], res.get('number'), flush=True)


def noissue_one(full: str, since: str) -> list[dict]:
    o, n = full.split('/')
    d = gh(['graphql', '-f', f'query={NOISSUE_Q}', '-f', f'o={o}', '-f', f'n={n}'])
    repo = ((d or {}).get('data') or {}).get('repository')
    if not repo:
        return [{'repo': full, 'error': 'none'}]
    rows = []
    for p in repo['pullRequests']['nodes']:
        mc = p.get('mergeCommit') or {}
        if not p['mergedAt'] or p['mergedAt'] < since or p['closingIssuesReferences']['totalCount'] or not mc:
            continue
        files = (p.get('files') or {}).get('nodes') or []
        src = [f for f in files if f['path'].endswith('.py') and not TEST.search(f['path'])]
        tst = [f for f in files if TEST.search(f['path']) and f['path'].endswith('.py')]
        body = (p.get('bodyText') or '').strip()
        if not (src and tst) or len(src) > 4 or sum(f['additions'] for f in src) > 150 or len(body) < 40:
            continue
        rows.append({'repo': full, 'number': p['number'], 'title': p['title'], 'url': p['url'],
                     'created_at': p['createdAt'], 'merged_at': p['mergedAt'], 'merge_commit': mc.get('oid'),
                     'parents': [x['oid'] for x in ((mc.get('parents') or {}).get('nodes') or [])],
                     'issue': {'title': p['title'], 'body': body, 'from_pr': True},
                     'src_files': [f['path'] for f in src], 'test_files': [f['path'] for f in tst]})
    return rows or [{'repo': full, 'none': True}]


def noissue(root: Path, repos_file: Path, since: str, workers: int) -> None:
    repos = repos_file.read_text(encoding='utf-8').split()
    out = root / 'noissue_details.jsonl'
    done = done_keys(out, lambda d: d['repo'])
    todo = [r for r in repos if r not in done]
    print('repos', len(repos), 'todo', len(todo), flush=True)
    with out.open('a', encoding='utf-8') as f, ThreadPoolExecutor(workers) as ex:
        for i, rows in enumerate(ex.map(lambda r: noissue_one(r, since), todo)):
            for r in rows:
                f.write(json.dumps(r) + '\n')
            f.flush()
            if i % 50 == 0:
                print(i, rows[0]['repo'], sum(1 for r in rows if 'number' in r), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['search', 'prs', 'details', 'all', 'noissue'])
    ap.add_argument('--repos', type=Path, default=Path('data/mine3/mining/repos_train.txt'))
    ap.add_argument('--root', type=Path, default=Path('data/mine3/mining'))
    ap.add_argument('--since', default='2024-01-01', help='oldest merge date kept (priv-eval later keeps 2025-07+)')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    if a.stage in ('search', 'all'):
        search(a.root)
    if a.stage in ('prs', 'all'):
        prs(a.root, a.since, a.workers)
    if a.stage in ('details', 'all'):
        details(a.root, a.workers)
    if a.stage == 'noissue':
        noissue(a.root, a.repos, a.since, a.workers)
    print('STAGE-DONE', a.stage, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
