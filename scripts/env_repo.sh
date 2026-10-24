#!/usr/bin/env bash
# D-053 funnel fix: per-repo wheelhouse for grading in the scorer's way (the harness installs every wheel in
# /wheels, then imports the repo from source). For each task snapshot of the repo (newest first), inside the
# official sandbox image (Python 3.13) WITH network: collect every declared requirement (deps_list.py), build
# wheels with `pip wheel` (sdists become wheels); when the bulk resolve fails, again with every pin relaxed, then
# one requirement at a time, and
# a pin that has no 3.13 wheel falls back to the bare name. Wheels of the package under test are removed (the
# published release would shadow the source). A shared pip cache makes later repos fast.
# Usage: scripts/mining/env_repo.sh <root (tasks.jsonl, snapshots/)> <repo prefix>   e.g. data/mine3 narwhals
#   writes <root>/wheels_by_repo/<repo>/*.whl and <root>/wheels_by_repo/<repo>/.env.log
set -u
ROOT=${1:?root}; repo=${2:?repo}
IMG=${IMG:-swebench-sandbox:latest}; MAXT=${MAXT:-1}
out="$ROOT/wheels_by_repo/$repo"; mkdir -p "$out" .local/cache/pip
[ -f "$out/.done" ] && { echo "$repo: env done earlier"; exit 0; }
win() { cygpath -w "$(realpath "$1")" 2>/dev/null || realpath "$1"; }
ids=$(ls "$ROOT/snapshots" | sed -n "s/^\(${repo}_[0-9]*\)\.tgz$/\1/p" | sort -t_ -k2 -nr | head -n "$MAXT")
[ -n "$ids" ] || { echo "$repo: no snapshots"; exit 0; }
for iid in $ids; do
  echo "$(date +%H:%M:%S) $repo env from $iid" >> "$out/.env.log"
  MSYS_NO_PATHCONV=1 timeout 900 docker run --rm --memory 3g --cpus 2 \
    -v "$(win "$ROOT/snapshots/$iid.tgz"):/snap.tgz:ro" -v "$(win "$out"):/out" \
    -v "$(win .local/cache/pip):/root/.cache/pip" -v "$(win scripts/mining/deps_list.py):/deps_list.py:ro" \
    "$IMG" bash -c '
      set -u; mkdir -p /ws && tar xzf /snap.tgz -C /ws 2>/dev/null; cd /ws
      python3 /deps_list.py /ws > /tmp/reqs.txt; echo "  $(wc -l < /tmp/reqs.txt) requirements"
      sed -E "s/^([A-Za-z0-9._-]+(\[[^]]*\])?).*/\1/" /tmp/reqs.txt > /tmp/relaxed.txt
      if pip wheel -q -w /out --prefer-binary -r /tmp/reqs.txt 2>/tmp/err.txt; then echo "  bulk ok"
      elif pip wheel -q -w /out --prefer-binary -r /tmp/relaxed.txt 2>/tmp/err2.txt; then
        echo "  bulk ok with pins relaxed ($(grep -m1 -E "ERROR" /tmp/err.txt | cut -c1-120))"
      else
        echo "  bulk failed: $(grep -m1 -E "ERROR" /tmp/err.txt | cut -c1-160)"
        while read -r r; do
          [ -n "$r" ] || continue
          pip wheel -q -w /out --prefer-binary "$r" 2>/dev/null && continue
          n=$(printf "%s" "$r" | sed -E "s/^([A-Za-z0-9._-]+).*/\1/")
          if pip wheel -q -w /out --prefer-binary "$n" 2>/dev/null; then echo "  unpinned: $r -> $n"; else echo "  MISSING: $r"; fi
        done < /tmp/reqs.txt
      fi
      python3 /deps_list.py /ws --own > /tmp/own.txt
      for f in /out/*.whl; do
        b=$(basename "$f" | cut -d- -f1 | tr "A-Z." "a-z_")
        if grep -qx "$b" /tmp/own.txt; then echo "  removed own wheel $(basename "$f")"; rm -f "$f"; fi
      done' >> "$out/.env.log" 2>&1 || echo "  FAIL/TIMEOUT $iid" >> "$out/.env.log"
done
n=$(ls "$out" | grep -c '\.whl$'); echo "$repo: $n wheels, $(du -sh "$out" | cut -f1)" | tee -a "$out/.env.log"; touch "$out/.done"
