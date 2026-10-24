# Scratch step 2: per candidate repo, language mix + merged PRs (since 2025-07) that touch src .py + tests and close an issue
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

Q = '''query($o:String!,$n:String!){repository(owner:$o,name:$n){
 isFork primaryLanguage{name} languages(first:8,orderBy:{field:SIZE,direction:DESC}){totalSize edges{size node{name}}}
 root: object(expression:"HEAD:"){... on Tree{entries{name type}}}
 pullRequests(states:MERGED,last:60,orderBy:{field:UPDATED_AT,direction:ASC}){nodes{
   number title mergedAt baseRefName additions deletions changedFiles bodyText
   closingIssuesReferences(first:2){nodes{number title createdAt bodyText}}
   files(first:40){nodes{path additions deletions changeType}}}}}}'''
TEST = re.compile(r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$')
def one(full):
    o, n = full.split('/')
    for attempt in range(3):
        r = subprocess.run(['gh','api','graphql','-f',f'query={Q}','-f',f'o={o}','-f',f'n={n}'], capture_output=True, text=True, encoding='utf-8')
        if r.returncode == 0: break
        time.sleep(5 * (attempt + 1))
    else: return {'repo': full, 'error': r.stderr[:200]}
    d = json.loads(r.stdout)['data']['repository']
    if not d: return {'repo': full, 'error': 'none'}
    tot = d['languages']['totalSize'] or 1
    py = sum(e['size'] for e in d['languages']['edges'] if e['node']['name'] == 'Python') / tot
    native = sum(e['size'] for e in d['languages']['edges'] if e['node']['name'] in ('C','C++','Rust','Cython','Go','Fortran')) / tot
    root = {e['name'] for e in ((d.get('root') or {}).get('entries') or [])}
    prs = []
    for p in d['pullRequests']['nodes']:
        if not p['mergedAt'] or p['mergedAt'] < '2025-07-01': continue
        files = (p.get('files') or {}).get('nodes') or []; src = [f for f in files if f['path'].endswith('.py') and not TEST.search(f['path'])]
        tst = [f for f in files if TEST.search(f['path']) and f['path'].endswith('.py')]
        issues = (p.get('closingIssuesReferences') or {}).get('nodes') or []
        if not (src and tst and issues): continue
        add = sum(f['additions'] for f in src); dele = sum(f['deletions'] for f in src)
        prs.append({'number': p['number'], 'title': p['title'], 'merged': p['mergedAt'], 'base': p['baseRefName'], 'src_files': len(src), 'test_files': len(tst),
                    'src_add': add, 'src_del': dele, 'issue': issues[0]['number'], 'issue_title': issues[0]['title'], 'issue_chars': len(issues[0]['bodyText'] or '') + len(issues[0]['title'])})
    return {'repo': full, 'py': round(py, 3), 'native': round(native, 3), 'pyproject': bool(root & {'pyproject.toml','setup.py','setup.cfg'}),
            'tests_dir': bool(root & {'tests','test'}), 'root': sorted(root)[:40], 'prs': prs}
repos = [json.loads(l)['full_name'] for l in open('repos.jsonl', encoding='utf-8')]
done = set()
try: done = {json.loads(l)['repo'] for l in open('prs.jsonl', encoding='utf-8')}
except FileNotFoundError: pass
todo = [r for r in repos if r not in done]; print('todo', len(todo), flush=True)
with open('prs.jsonl', 'a', encoding='utf-8') as out, ThreadPoolExecutor(4) as ex:
    for i, res in enumerate(ex.map(one, todo)):
        out.write(json.dumps(res) + '\n'); out.flush()
        if i % 50 == 0: print(i, res['repo'], len(res.get('prs', [])), flush=True)
print('done', flush=True)
