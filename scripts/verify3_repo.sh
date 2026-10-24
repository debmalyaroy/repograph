#!/usr/bin/env bash
# D-053: verify one repo of a mined batch the way pub129 was declared sound (gold resolves, empty does not),
# in the scorer's sandbox (official harness via scripts/grade.py, wheels installed from /wheels).
# Steps: per-repo wheelhouse (env_repo.sh) -> VCS version-file stubs (fix_snapshots.py) -> gold grade ->
# empty grade for the gold-resolved tasks -> <root>/verify/by_repo/<repo>.sound. A repo with no sound task
# loses its harness cache and wheels (disk hygiene); sound repos keep both for later grading.
# Usage: scripts/mining/verify3_repo.sh <root> <repo prefix>     (resumable: skips repos with a .sound file)
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
ROOT=${1:?root}; repo=${2:?repo}; V=$ROOT/verify; W=$ROOT/wheels_by_repo/$repo
mkdir -p "$V/by_repo" "$V/gold" "$V/empty" .local/logs
[ -f "$ROOT/sandbox/setup.py" ] || { mkdir -p "$ROOT/sandbox"; cp data/sandbox/setup.py "$ROOT/sandbox/"; }
[ -f "$V/by_repo/$repo.sound" ] && { echo "$repo: verified earlier ($(wc -l < "$V/by_repo/$repo.sound") sound)"; exit 0; }
ids="$V/by_repo/$repo.ids"
ls "$ROOT/snapshots" | sed -n "s/^\(${repo}_[0-9]*\)\.tgz$/\1/p" > "$ids"
[ -s "$ids" ] || { echo "$repo: no snapshots"; exit 0; }
log=".local/logs/verify3-$(basename "$ROOT")-$repo.log"
{
  echo "$(date +%H:%M:%S) $repo: $(wc -l < "$ids") tasks"
  bash scripts/mining/env_repo.sh "$ROOT" "$repo"
  uv run python scripts/mining/fix_snapshots.py "$ROOT/snapshots" --ids "$ids"
  uv run python - "$ROOT" "$ids" <<'PY'
import json, sys
from pathlib import Path
root, ids = Path(sys.argv[1]), set(Path(sys.argv[2]).read_text().split())
for line in (root / 'tasks.jsonl').open(encoding='utf-8'):
    t = json.loads(line)
    if t['instance_id'] in ids:
        (root / 'verify/gold' / f"{t['instance_id']}.patch").write_text(t['patch'], encoding='utf-8', newline='')
        (root / 'verify/empty' / f"{t['instance_id']}.patch").write_text('', encoding='utf-8')
PY
} >> "$log" 2>&1
if [ "$(ls "$W" 2>/dev/null | grep -c '\.whl$')" -eq 0 ]; then
  echo "$repo: no wheels" | tee -a "$log"; : > "$V/by_repo/$repo.sound"; exit 0
fi
grade() {
  uv run python scripts/grade.py --tasks "$ROOT/tasks.jsonl" --snapshots "$ROOT/snapshots" --wheels "$W" \
    --cache-root .local/cache/sp --no-pin --ids-file "$2" --patches-dir "$V/$1" \
    --out "$V/by_repo/${repo}_$1.jsonl" --concurrency 2 >> "$log" 2>&1
}
rm -f "$V/by_repo/${repo}_gold.jsonl" "$V/by_repo/${repo}_empty.jsonl"
grade gold "$ids"
ok="$V/by_repo/$repo.gold_ok"
grep -o '"instance_id": "[^"]*", "resolved": true' "$V/by_repo/${repo}_gold.jsonl" 2>/dev/null | cut -d'"' -f4 > "$ok"
: > "$V/by_repo/$repo.sound.tmp"
if [ -s "$ok" ]; then
  grade empty "$ok"
  grep -o '"instance_id": "[^"]*", "resolved": false' "$V/by_repo/${repo}_empty.jsonl" 2>/dev/null | cut -d'"' -f4 \
    | grep -Fxf "$ok" > "$V/by_repo/$repo.sound.tmp"
fi
mv "$V/by_repo/$repo.sound.tmp" "$V/by_repo/$repo.sound"
n=$(wc -l < "$V/by_repo/$repo.sound")
if [ "$n" -eq 0 ]; then
  fp=$(ls "$W" | grep '\.whl$' | LC_ALL=C sort | awk 'NR>1{printf "\n"}{printf "%s",$0}' | sha256sum | cut -c1-16)
  rm -rf ".local/cache/sp/$fp" "$W"
fi
echo "$(date +%H:%M:%S) $repo: gold $(wc -l < "$ok")/$(wc -l < "$ids"), sound $n" | tee -a "$log"
