#!/usr/bin/env bash
# C3: shallow clones (HEAD) of the pure-Python packaged candidate repos for the Atlas / `similar` index.
# Resumable: skips repos already cloned. Log: .local/logs/atlas-clone.log
cd /d/codebase/gemma-agent || exit 1
mkdir -p data/atlas/repos
python -c "
import json
for l in open('.local/tmp/priv60/prs.jsonl', encoding='utf-8'):
    x = json.loads(l)
    if 'prs' in x and x['py'] >= 0.85 and x['native'] < 0.05 and x['pyproject']:
        print(x['repo'])
" | tr -d '\r' > .local/tmp/priv60/atlas_repos.txt
echo "repos: $(wc -l < .local/tmp/priv60/atlas_repos.txt)"
n=0
while read -r repo; do
  dst="data/atlas/repos/${repo//\//__}"
  if [ -d "$dst/.git" ]; then continue; fi
  n=$((n+1))
  timeout 300 git clone -q --depth 1 --filter=blob:limit=2m "https://github.com/$repo.git" "$dst" 2>&1 | tail -1
  echo "$(date +%H:%M:%S) $n $repo $([ -d "$dst/.git" ] && echo ok || echo FAIL)"
done < .local/tmp/priv60/atlas_repos.txt
echo "done $(date)"
