#!/usr/bin/env bash
# Scratch step 1: candidate pure-Python repos, low profile, active 2025-26 (GitHub search, read-only)
out=repos.jsonl; : > $out
for stars in 50..150 150..400 400..1000 1000..3000; do
  for topic in "" "topic:cli" "topic:library" "topic:api" "topic:parser" "topic:http" "topic:testing" "topic:data"; do
    q="language:python stars:$stars pushed:>2026-03-01 created:2018-01-01..2025-06-01 archived:false fork:false $topic"
    gh api -X GET search/repositories -f q="$q" -f sort=updated -f per_page=100 \
      --jq '.items[] | {full_name, stars: .stargazers_count, size, created_at, pushed_at, topics, description}' >> $out 2>>err.log
    sleep 2.5
  done
done
sort -u $out -o $out; wc -l $out
