#!/usr/bin/env bash
# D-053: verify every repo of a mined batch, P repos in parallel (default 2: WSL is capped at 8 GB / 8 CPUs),
# then write <root>/sound.txt and a one-line summary. Resumable (verify3_repo.sh skips finished repos).
# Usage: [P=2] [REPOS=file] bash scripts/mining/verify3.sh <root>       e.g. data/mine3/eval
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=${1:?root}; P=${P:-2}
if [ -n "${REPOS:-}" ]; then tr -d '\r' < "$REPOS"; else ls "$ROOT/snapshots" | sed -E 's/_[0-9]+\.tgz$//' | sort -u; fi \
  | xargs -P "$P" -I{} bash scripts/mining/verify3_repo.sh "$ROOT" {}
cat "$ROOT"/verify/by_repo/*.sound 2>/dev/null | sort > "$ROOT/sound.txt"
echo "VERIFY3-DONE $ROOT: $(cat "$ROOT"/verify/by_repo/*.ids | wc -l) tasks, $(cat "$ROOT"/verify/by_repo/*.gold_ok 2>/dev/null | wc -l) gold ok, $(wc -l < "$ROOT/sound.txt") sound"
