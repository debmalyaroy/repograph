# Scratch C5 selection: priv60 repos = Atlas repos most structurally similar to the public repos
# (repograph fingerprints, z-scored over all graphs), in three size bands like the public set
# (requests-like small, rich-like medium, fastapi-like large), with mined candidate PRs.
# Picks ~10 repos per band (spares: gold verification drops some), writes selected_repos.txt.
import collections
import json
import math

F = [json.loads(l) for l in open('.local/tmp/priv60/fingerprints.jsonl')]
d = len(F[0]['raw']); n = len(F)
mu = [sum(f['raw'][j] for f in F) / n for j in range(d)]
sd = [math.sqrt(sum((f['raw'][j] - mu[j]) ** 2 for f in F) / n) + 1e-9 for j in range(d)]
Z = {f['name']: [(f['raw'][j] - mu[j]) / sd[j] for j in range(d)] for f in F}
nodes = {f['name']: math.exp(f['raw'][0]) for f in F}
cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-12)
public = {k: v for k, v in Z.items() if k.split('_')[0] in ('fastapi', 'rich', 'requests', 'httpx') and len(k.rsplit('_', 1)[-1]) == 40}
details = [json.loads(l) for l in open('.local/tmp/priv60/pr_details.jsonl', encoding='utf-8')]
ok = [x for x in details if x.get('merge_commit') and x.get('issue') and x.get('parents')]
prs = collections.Counter(x['repo'].replace('/', '__') for x in ok)
rows = []
for k, v in Z.items():
    if k in public or prs.get(k, 0) == 0:
        continue
    sims = sorted((cos(v, pv), pk.split('_')[0]) for pk, pv in public.items())
    best = sims[-1]
    rows.append({'repo': k.replace('__', '/', 1), 'sim': round(best[0], 3), 'like': best[1], 'nodes': int(nodes[k]), 'prs': prs[k]})
bands = [('small', 0, 1200), ('medium', 1200, 3000), ('large', 3000, 10 ** 9)]
sel = []
for name, lo, hi in bands:
    cand = sorted([r for r in rows if lo <= r['nodes'] < hi], key=lambda r: -r['sim'])
    picked, tasks = [], 0
    for r in cand:
        if len(picked) >= 10:
            break
        picked.append(r)
        tasks += min(5, r['prs'])
    print(f"== {name}: {len(cand)} candidates, picked {len(picked)} repos, up to {tasks} tasks")
    for r in picked:
        print(f"   {r['sim']:.3f} ~{r['like']:8} {r['repo']:45} nodes {r['nodes']:6} PRs {r['prs']}")
    sel += picked
open('.local/tmp/priv60/selected_repos.txt', 'w').write('\n'.join(r['repo'] for r in sel) + '\n')
json.dump(sel, open('.local/tmp/priv60/selected.json', 'w'), indent=1)
print('selected', len(sel), 'repos')
