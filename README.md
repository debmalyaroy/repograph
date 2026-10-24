# repograph

Competition-schema code graphs for Python repositories, and the tools that turn real GitHub pull requests
into verified SWE-style tasks.

`repograph` was built for the [Gemma 4 Developer Agent](https://www.kaggle.com/competitions/gemma-4-developer-agent)
competition, which ships code graphs and 256-d node embeddings for its public repositories but no generator.
It:

- **rebuilds the official graphs from source** for any Python repository (`build`, `build-dir`), with node
  precision/recall 0.998 and edge precision/recall 0.909/0.958 against all 127 official graphs (`parity`);
- **approximates the official embeddings** for repositories that have none (`embed`): the official vectors
  are structural (graph neighbours cosine 0.99, random pairs 0.77–0.90), so hashed AST/token features
  propagated over the graph and mapped by a ridge fit reproduce them far better than any text encoder;
- **ranks repositories by structural similarity** (`similar`) with a 19-feature graph fingerprint;
- **mines and packages tasks** in the exact competition format (`mine`, `snapshots`, `normalize`, `check`,
  `depcheck`), the pipeline behind the public dataset
  [repograph-verified-swe-tasks](https://www.kaggle.com/datasets/debmalya84/repograph-verified-swe-tasks)
  (221 verified private-like evaluation tasks from 102 repositories, plus training splits).

One static binary, no Python dependency for the graph work; the mining and verification helpers in
`scripts/` are Python (uv-runnable) and need Docker for the grader.

## Install

Prebuilt binaries are attached to each [release](https://github.com/debmalyaroy/repograph/releases)
(`repograph-x86_64-unknown-linux-musl`, `repograph-x86_64-pc-windows-msvc.exe`). Or build from source:

```bash
cargo install --git https://github.com/debmalyaroy/repograph   # Rust 1.85+ (edition 2024)
# static Linux binary:
docker build -f Dockerfile.musl -t repograph-musl . && docker run --rm repograph-musl cat /out/repograph > repograph && chmod +x repograph
```

The optional `embed-probe` command (compare open text encoders with the official vectors) needs
`--features textembed` (fastembed + onnxruntime); everything else builds without it.

## The graph schema

The official files are NetworkX node-link JSON: `{"directed": false, "multigraph": true, "graph": {...},
"nodes": [{"id": "pkg.module.Class.method", "name": "method", "text": "<source>"}], "edges": [{"type": "calls",
"source": ..., "target": ..., "key": 0}]}`, one file per `<repo>_<commit>`. Rules recovered by diffing against
the official graphs (see [docs/parity.md](docs/parity.md)):

| Rule | Official generator | `repograph build` |
|---|---|---|
| Module path | from the repo root, a leading `src/` dropped | same |
| Nodes | `def` and `class` at module or class level; no `async def`; nothing inside a function | same (parity mode); our agent's engine additionally indexes `async def`, which the official graphs omit |
| Nested classes | one "current class" variable, not a stack | same (reproduces the official quirk) |
| Edges | undirected class↔member, base↔subclass, and calls resolved through imports, then bare name or `Receiver.method` | same |
| Function text | source from `def` | 98.4 % exact |
| Class text | `ast.unparse`-style summary | 62 % exact (known gap) |

## Commands

```text
repograph build <repo dir | snapshot.tgz> --out graph.json [--name <repo>]
repograph build-dir <root> --out <dir>          one graph per repository directory, in parallel
repograph parity [--tasks tasks.jsonl --snapshots dir --graphs dir]    rebuild + compare every official graph
repograph embed <graphs...> --out <dir> [--train keys... --dim 1024 --hops 5 --lambda 10]
repograph similar --index <graph dirs> --target <stem> [--k 10] [--validate] [--dump fp.jsonl]
repograph mine --details pr_details.jsonl --out <dir> --clones <dir> [--repos file --per-repo 5]
repograph snapshots tasks.jsonl --out <dir> [--ids file --threads 8]
repograph normalize tasks.jsonl --out tasks.jsonl --meta meta.jsonl [--source name]
repograph check tasks.jsonl [--snapshots dir --reference pub129.jsonl --ok-ids file --skip-head]
repograph depcheck <repo dirs | snapshots...> | --tasks tasks.jsonl --snapshots dir
```

- `build` / `build-dir`: tree-sitter parse of every tracked `.py` file in parallel (about 25 ms per 1,000
  files), import resolution, call and inheritance edges.
- `parity`: for each official task commit, unpack the snapshot, rebuild, and report node/edge precision and
  recall plus text agreement (JSONL, one line per graph).
- `embed`: leave-one-repo-out fidelity report, then `.npz` files (`<node id> -> 256 floats`) in the official
  space for graphs that have no official vectors. Subtract the repository mean before any cosine search:
  the official and the approximated vectors are anisotropic (all pairs look similar); after mean-centering,
  random pairs sit at 0.04–0.13 and graph neighbours at 0.74–0.94.
- `similar`: z-scored cosine over a 19-feature fingerprint (size, density, degree profile, class/method/test
  shares, module depth, function length, inheritance share). `--validate` checks that every official
  commit's nearest neighbour is another commit of the same repository (126/126).
- `mine`: from a PR-details JSONL (`repo, number, merge_commit, parents, issue, created_at`) build
  `tasks.jsonl` rows (base = merge-commit parent, issue text as `problem_statement`, source diff as `patch`,
  test diff as `test_patch`), a frozen LF snapshot at the base commit (50 commits of history, nothing after),
  and the graph.
- `check`: the format gate used on every batch: exactly the eight official fields in order, plain
  `--- a/ +++ b/` hunks, no test paths in `patch`, only test paths in `test_patch`, snapshot tree = base
  commit; prints the batch's patch-size and issue-length quantiles next to the reference set's.
- `depcheck`: which test imports, fixtures and pytest plugins the project does not declare (the usual reason
  a gold patch fails in a clean environment).

## Building a verified task set (the dataset recipe)

```bash
# 1. candidate repositories and PRs (GitHub API, read-only; resumable)
uv run python scripts/mine3.py search  --root work/mining
uv run python scripts/mine3.py prs     --root work/mining --since 2024-01-01 --workers 12
uv run python scripts/mine3.py details --root work/mining --workers 12
uv run python scripts/split3.py                       # repo-disjoint eval / train PR lists
# 2. tasks, snapshots, graphs
repograph mine --details work/mining/details_eval.jsonl --out work/eval --clones work/clones --per-repo 3
# 3. format gate
repograph check work/eval/tasks.jsonl --ok-ids work/eval/format_ok.txt
# 4. environments + soundness in the competition's own grader (Docker)
bash scripts/env_repo.sh work/eval <repo>            # per-repo wheelhouse from every declared dependency source
P=2 bash scripts/verify3.sh work/eval                 # gold passes and empty fails -> work/eval/sound.txt
# 5. licences and assembly
uv run python scripts/licenses.py work/eval
uv run python scripts/assemble.py                     # release/<split>/{tasks,meta}.jsonl + snapshots + graphs + wheels
```

`scripts/grade.py` wraps the competition harness's `verify_task` (resets test and config paths the patch
touches, applies the hidden test patch, runs pytest with a 240 s timeout, parses JUnit). A task is sound only
if the gold patch makes every test in the touched test files pass and the empty patch does not.

## Numbers behind the claims

| Measurement | Value | Where |
|---|---|---|
| Node precision / recall vs 127 official graphs | 0.998 / 0.998 | `parity` |
| Edge precision / recall | 0.909 / 0.958 | `parity` |
| Function text exact / class summary exact | 98.4 % / 62 % | `parity` |
| Official embeddings: random-pair cosine (requests / rich / fastapi) | 0.863 / 0.896 / 0.770 | `docs/embeddings.md` |
| Official embeddings: 1-hop neighbour cosine | 0.990 / 0.985 / 0.990 | same |
| After repo-mean centering: random / neighbour (fastapi) | 0.128 / 0.941 | same |
| Text encoder (jina-v2-base-code) linear map, held-out top-10 agreement | 5 % | `embed-probe` |
| Hashed tokens propagated 5–8 hops, held-out top-10 agreement | 32 % | `embed` |
| Repository fingerprint gate (same-repo nearest neighbour) | 126 / 126 | `similar --validate` |

## Licence

Apache-2.0. The tasks in the companion dataset keep their source repositories' licences (recorded per task).
