# Scratch C5 step 1: details for the in-band PRs (merge commit + parents, issue text, dates, file list)
# so tasks can be built in the competition format. Output: .local/tmp/priv60/pr_details.jsonl
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

TEST = re.compile(r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$')
Q = '''query($o:String!,$n:String!,$num:Int!){repository(owner:$o,name:$n){pullRequest(number:$num){
 number title url createdAt mergedAt baseRefName mergeCommit{oid parents(first:2){nodes{oid}}}
 commits(first:1){totalCount} additions deletions
 closingIssuesReferences(first:1){nodes{number title body createdAt url}}
 files(first:60){nodes{path additions deletions changeType}}}}}'''


def one(job):
    repo, num = job
    o, n = repo.split('/')
    for attempt in range(3):
        r = subprocess.run(['gh', 'api', 'graphql', '-f', f'query={Q}', '-f', f'o={o}', '-f', f'n={n}', '-F', f'num={num}'],
                           capture_output=True, text=True, encoding='utf-8')
        if r.returncode == 0:
            break
        time.sleep(5 * (attempt + 1))
    else:
        return {'repo': repo, 'number': num, 'error': r.stderr[:200]}
    pr = json.loads(r.stdout)['data']['repository']['pullRequest']
    files = (pr.get('files') or {}).get('nodes') or []
    iss = ((pr.get('closingIssuesReferences') or {}).get('nodes') or [None])[0]
    mc = pr.get('mergeCommit') or {}
    parents = [p['oid'] for p in ((mc.get('parents') or {}).get('nodes') or [])]
    return {'repo': repo, 'number': num, 'title': pr['title'], 'url': pr['url'], 'created_at': pr['createdAt'], 'merged_at': pr['mergedAt'],
            'merge_commit': mc.get('oid'), 'parents': parents, 'n_commits': (pr.get('commits') or {}).get('totalCount'),
            'issue': iss, 'src_files': [f['path'] for f in files if f['path'].endswith('.py') and not TEST.search(f['path'])],
            'test_files': [f['path'] for f in files if TEST.search(f['path'])],
            'other_files': [f['path'] for f in files if not f['path'].endswith('.py') and not TEST.search(f['path'])]}


jobs = []
for l in open('.local/tmp/priv60/prs.jsonl', encoding='utf-8'):
    x = json.loads(l)
    if 'prs' in x and x['py'] >= 0.85 and x['native'] < 0.05 and x['pyproject']:
        for p in x['prs']:
            if p['src_files'] <= 4 and p['src_add'] <= 120:
                jobs.append((x['repo'], p['number']))
done = set()
try:
    done = {(d['repo'], d['number']) for d in map(json.loads, open('.local/tmp/priv60/pr_details.jsonl', encoding='utf-8'))}
except FileNotFoundError:
    pass
todo = [j for j in jobs if j not in done]
print('jobs', len(jobs), 'todo', len(todo), flush=True)
with open('.local/tmp/priv60/pr_details.jsonl', 'a', encoding='utf-8') as out, ThreadPoolExecutor(4) as ex:
    for i, res in enumerate(ex.map(one, todo)):
        out.write(json.dumps(res) + '\n')
        out.flush()
        if i % 100 == 0:
            print(i, res['repo'], res.get('number'), flush=True)
print('done', flush=True)
