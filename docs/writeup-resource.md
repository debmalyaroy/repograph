## Abstract

Software-engineering agents are usually judged on a handful of famous repositories, and the tools that help them read code, such as call graphs and code embeddings, are rarely available for anything else. This writeup presents two open resources that remove both limits for Python. **repograph** is a Rust tool that builds code graphs in the exact schema of the Gemma 4 Developer Agent competition for any repository, approximates the competition's node embeddings, ranks repositories by structural similarity, and turns merged pull requests into verifiable tasks. **SWE Tasks** is a public dataset built with it: 221 GitHub-issue tasks from 102 Python repositories outside the usual benchmarks, each with the repository snapshot before the fix, the gold patch, the hidden tests, a code graph and a licence record, and each verified in the competition's own grader so that the gold patch passes and the empty patch fails. A companion dataset ships offline environments for all 108 repositories, and two public notebooks show how to explore the data, run a Gemma 4 model on a task and verify a patch the way the scorer does. The graph generator reaches 0.998 node and 0.909/0.958 edge precision/recall against all 127 official graphs; the analysis of the official embeddings shows they encode graph position rather than text, and that subtracting the repository mean turns them into a usable similarity space.

## 1. Introduction

Two gaps make it hard to develop or evaluate a coding agent beyond a competition's public data. The first is coverage: public SWE-style benchmarks draw on a small set of well-known repositories, so any score on them says little about the long tail of ordinary projects where most software work happens, and models may have seen the famous code during pre-training. The second is tooling: the Gemma 4 Developer Agent competition gives agents a code graph and 256-dimensional node embeddings for each repository, but the generator that produces them is not published. An agent that learns to use these graphs can be run only where the organisers pre-computed them.

The resources described here close both gaps. repograph reproduces the graph generator from its outputs, documents the rules it recovered, and adds what a graph-aware agent needs beyond the official files: embeddings for unseen repositories, a similarity index over repositories, and a pipeline that mines and verifies tasks in the competition's format. SWE Tasks is the first dataset produced by that pipeline. Because its task records have exactly the eight fields of the competition's public `tasks.jsonl`, any harness, agent or training script written for the public tasks runs on it without change.

Everything is public: the tool and its scripts on GitHub under Apache-2.0, the dataset and the environments on Kaggle, and the notebooks that demonstrate them. Each task keeps the licence of the repository it comes from, recorded per task.

## 2. The tool: repograph

repograph is a single static binary (Linux and Windows builds are on the release page) with subcommands that cover the life cycle of a code-graph resource.

**Graph generation (`build`, `build-dir`, `parity`).** The competition's graphs are NetworkX node-link JSON: one node per `def` or `class` with its source text, and edges for calls, class membership and inheritance. The rules the official generator follows were recovered by rebuilding the graph of every public task's snapshot and diffing against the official file, over all 127 distinct commits. The recovered rules are: module paths are taken from the repository root with a leading `src/` directory dropped; nodes are created for `def` and `class` at module or class level only, so `async def` and anything nested inside a function are omitted; nested classes are tracked with a single "current class" variable rather than a stack; call edges are resolved through imports first, then by bare name or `Receiver.method`, never through `self`; function text is the source from `def`, class text is an unparsed summary. Table 1 gives the agreement. `parity` re-runs this comparison on demand, so the generator can be re-validated whenever the official data changes.

| Table 1. Agreement with the 127 official graphs | Value |
|---|---|
| Node precision / recall | 0.998 / 0.998 |
| Edge precision / recall | 0.909 / 0.958 |
| Function text identical | 98.4 % |
| Class summary identical | 62 % |

The weakest repository is requests (edge precision 0.80), where a few call edges and the class-summary text still differ; both are listed as known gaps. Parsing uses tree-sitter in parallel and takes about 25 ms per 1,000 files, so a graph for a large repository is built in well under a second inside a task sandbox.

**Embeddings (`embed`).** The official 256-dimensional node vectors are not text embeddings. Across the three public repositories, random node pairs have a mean cosine of 0.86, 0.90 and 0.77, graph neighbours 0.99, two-hop neighbours 0.98 and same-module nodes 0.89 to 0.97 (Table 2). A linear map from an open code-text encoder reproduces only 5 % of the official top-10 neighbourhoods on a held-out repository, while hashed structural tokens propagated five to eight hops over the rebuilt graph reproduce 32 %. The vectors therefore encode position in the graph. `embed` produces vectors in the same space for repositories that have none, by propagating hashed AST-shape and identifier features over the graph and fitting a ridge map to the official vectors.

| Table 2. Mean cosine of official vectors (requests / rich / fastapi) | Random | 1 hop | 2 hops | Same module |
|---|---|---|---|---|
| raw | 0.863 / 0.896 / 0.770 | 0.990 / 0.985 / 0.990 | 0.985 / 0.976 / 0.988 | 0.972 / 0.974 / 0.886 |
| after subtracting the repository mean | 0.03 / 0.04 / 0.13 | 0.80 / 0.74 / 0.94 | | |

The second row is the practical finding: both the official and the approximated vectors are anisotropic, so a nearest-neighbour query in the raw space returns "anything in the same module". Subtracting the per-repository mean vector and renormalising moves random pairs to near zero while graph neighbours stay high (Figure 1), which makes the vectors usable for finding structural siblings, for example the parallel implementation of a method in a sister class.

![Figure 1. Cosine distributions of the official node embeddings for random pairs (red) and graph neighbours (blue/green), raw and after subtracting the repository mean.](https://raw.githubusercontent.com/debmalyaroy/repograph/main/docs/figures/f5-cosine.png)

**Repository similarity (`similar`).** A 19-feature fingerprint per graph (size, density, degree profile, shares of classes, methods and tests, module depth, function length, inheritance share), z-scored and compared by cosine, ranks repositories by structural resemblance to a target. Its sanity check: for every official commit, the nearest neighbour is another commit of the same repository (126 of 126). This is how repositories are chosen to resemble a target distribution, and it is also a general way to pick held-out repositories that look like a deployment target.

**Task mining and verification (`mine`, `snapshots`, `normalize`, `check`, `depcheck`, and the scripts).** `mine` turns a list of merged pull requests into tasks: the base commit is the merge commit's parent, the linked issue is the problem statement, the source part of the diff is the gold patch and the test part is the hidden test patch; the snapshot is a git checkout at the base commit with line-feed endings and fifty commits of history, and the graph is built at the same time. `normalize` converts other SWE-style datasets into the same eight-field form, moving every extra field to a sidecar. `check` is the format gate run on every batch: field names and order, plain diff hunks, no test files in the patch and only test files in the test patch, snapshot tree equal to the base commit, and a report of the batch's patch-size and issue-length quantiles next to a reference set. `depcheck` lists test imports, fixtures and pytest plugins that a project does not declare, the usual reason a gold patch fails in a clean environment. The Python scripts in `scripts/` do the rest: searching GitHub for candidate repositories and pull requests, building a per-repository offline environment from every declared dependency source, running the competition harness's `verify_task` in Docker, recording licences and assembling the splits.

## 3. The dataset: SWE Tasks

**Contents.** 221 evaluation tasks (`priv_eval`) from 102 repositories, and a training split (`priv_train`, 36 tasks from 6 repositories, growing as verification continues). The two splits share no repository, and none of the competition's public repositories appears in either. Every task comes from a pull request merged in July 2025 or later that closes an issue and changes both source and tests; at most three tasks come from one repository, so no project dominates.

**Verification.** A task is shipped only if, in the competition's own grader running in Docker with the task's environment, the gold patch makes every test in the touched test files pass and the empty patch does not. The grader resets any test or configuration file a candidate patch touched, applies the hidden test patch, runs pytest on the touched test files with a 240-second limit and requires every test in them to pass. The same procedure is what the companion notebook reproduces, so the guarantee can be checked by anyone in a fresh environment.

**Format.** `tasks.jsonl` holds exactly the eight competition fields: `instance_id`, `repo`, `base_commit`, `problem_statement`, `hints_text` (empty), `patch`, `test_patch`, `created_at`. `meta.jsonl` carries the source round, the repository licence and its class; `licenses.jsonl` lists every task's SPDX identifier and the file it was read from. Snapshots are `snapshots/<instance_id>.tgz`; graphs are `graphs/<repo>_<commit>.json` in the official schema (Figure 2).

![Figure 2. Layout of the dataset and the format contract shared with the competition's public tasks.](https://raw.githubusercontent.com/debmalyaroy/repograph/main/docs/figures/g2-layout.png)

**Shape.** The tasks are larger than the competition's public ones (Table 3, Figure 3): the median gold patch is 27 lines across two files with a 1,400-character issue, against 12 lines, one file and 418 characters; 63 of the 221 gold patches add or delete a file, which none of the public tasks do. The set is therefore a harder and broader held-out benchmark, and the per-task shape in the sidecar lets a user slice it to any band.

| Table 3. Quantiles p25 / p50 / p75 / p90 | SWE Tasks | Competition public tasks |
|---|---|---|
| Gold patch lines | 10 / 27 / 57 / 111 | 4 / 12 / 44 / 127 |
| Files touched | 1 / 2 / 3 / 5 | 1 / 1 / 2 / 4 |
| Issue characters | 679 / 1,394 / 2,539 / 3,979 | 146 / 418 / 1,000 / 1,779 |

![Figure 3. Task shape of the dataset against the 129 public competition tasks.](https://raw.githubusercontent.com/debmalyaroy/repograph/main/docs/figures/g4-shape.png)

**Licences.** Source repositories are MIT (135 tasks), Apache-2.0 (76), GPL-3.0 (13), BSD-3-Clause (11), LGPL-3.0 (9), LGPL-2.1 (8), BSD-2-Clause (3) and Unlicense (1); 196 tasks are from permissively licensed repositories and 25 from copyleft ones, filterable through `license_class`. Repositories without a licence file are excluded. The compilation, graphs, documentation and tools are Apache-2.0.

**Companions.** `SWE Tasks: offline environments` holds one folder per repository (named by the repository's short name) with the wheels that install its test environment on Python 3.13, so grading needs no network and uses the exact versions used for verification. Two public notebooks accompany the data: a how-to notebook that loads a split, opens a task, builds its environment, grades gold and empty patches, builds supervised-fine-tuning pairs and shows how to extend the set; and a demo notebook that explores the dataset, runs Gemma 4 E2B on a task with graph-guided localization and verifies the resulting patch exactly as the scorer does.

## 4. Experiments and measurements

**Generator parity** is the first experiment and the one that makes everything else trustworthy: Table 1 is computed by `parity` over all 127 official commits, and the remaining differences are itemised rather than averaged away.

**Embedding analysis** (Table 2, Figure 1) uses 4,000 random pairs and 4,000 graph-neighbour pairs per repository on one official commit each of requests (754 nodes), rich (1,925) and fastapi (4,287). The held-out neighbourhood agreement (5 % for a text encoder with a linear map, 32 % for propagated structural tokens) uses the official vectors of two repositories for fitting and the third for testing.

**The mining funnel** (Figure 4) documents yield at each stage. Rounds one and two screened 1,174 candidate repositories, kept 707 pure-Python packages, found 1,619 merged issue-linked pull requests, 1,046 of them inside the public set's patch-size band, mined 92 tasks from 30 structurally similar repositories and verified 61 as sound. A second verification pass after fixing environment construction recovered 37 of the 57 that had failed: the failures were undeclared dependency sources (hatch hooks, PEP 735 groups), pins without a Python 3.13 wheel and version files that git does not track, not bad tasks. Round three widened the search to about 1,500 similar repositories and 5,000 pull requests; 399 evaluation tasks from 184 repositories passed the format gate and 123 were sound. The lesson for anyone building such a set is that environment construction, not task selection, is where most candidates are lost, which is why the verifier installs every declared dependency source, falls back one requirement at a time, infers pytest plugins from configuration and test imports, and ships the resulting wheelhouse.

![Figure 4. From GitHub to verified tasks: yield at each stage of the mining and verification funnel.](https://raw.githubusercontent.com/debmalyaroy/repograph/main/docs/figures/g1-mining-funnel.png)

**Format gate.** `check` passes 221 of 221 shipped tasks and 129 of 129 public tasks, and produces the shape comparison of Table 3.

**Why a held-out set of this kind is needed.** Across eight leaderboard submissions of one agent in this competition, scores on the four public repositories over-predicted the private leaderboard about three-fold; a run that resolved 24.8 % of the public tasks on a TPU resolved none of the eleven private-like tasks it reached. Selecting an agent on the public repositories overfits to them. A repository-disjoint, recent, unfamiliar set is the proxy a developer needs, and this dataset is built to be one.

**Baseline.** The demo notebook gives a reproducible floor: on `marko_242`, Gemma 4 E2B with one-hop graph localization and one shot does not resolve the task, while the empty patch fails and the gold patch passes in the same fresh environment, which reproduces the dataset's guarantee outside the pipeline that produced it. A per-model, per-architecture table on the full evaluation split is in progress and will be added here.

## 5. Using the resources

Load a split and a task in three lines, as in the notebooks:

```python
rows = [json.loads(l) for l in open("priv_eval/tasks.jsonl")]
tarfile.open(f"priv_eval/snapshots/{rows[0]['instance_id']}.tgz").extractall("work")
graph = json.load(open(f"priv_eval/graphs/{short_repo}_{rows[0]['base_commit']}.json"))
```

Evaluate an agent by handing it the problem statement and the snapshot, collecting `git diff` of the workspace, and grading with the harness's `verify_task` or the notebook's `verify()` function against the hidden test patch. Train on `priv_train` with a reward that is the same grading function, or build supervised pairs from issue, graph context and gold patch with the how-to notebook. Extend the set to new repositories with `mine3.py`, `repograph mine`, `check` and `verify3.sh`; convert another SWE-style dataset with `normalize` and `snapshots`. Build the graph of any Python repository with `repograph build`, and its embeddings with `embed`, to give a graph-aware agent the same tools outside the competition.

## 6. Generality and impact

Nothing in the tool is tied to the competition except the schema it reproduces, and that schema is a plain, documented node-link graph. The generator works on any Python repository; the embedding analysis and the mean-centering correction apply to any anisotropic code-embedding space; the fingerprint similarity is a general way to pick held-out repositories; the mining pipeline accepts any list of pull requests and the format gate any eight-field task file. For agentic software engineering, the resource offers what has been missing between public leaderboards and private deployments: a verified, repository-disjoint, recent evaluation set with offline environments, and the means to build another one for a different target.

## 7. Limitations

The graph generator still differs from the official one in class-summary text and in a few call edges; the embeddings approximate a model whose training is unknown; soundness is relative to the shipped environments and Python 3.13; the tasks are public code chosen to resemble private code, not private code; the training split is small until its verification completes; and the baseline is a single-shot floor, not an agent result.

## 8. Related work

SWE-bench and SWE-bench Verified [1, 8] define the issue-to-patch task and the gold-and-tests grading used here. SWE-Gym [5], SWE-smith [6] and SWE-rebench [7] scale task collection with automated environment construction and address contamination with recent tasks; `normalize` converts their records into the competition format. R2E-Gym [19c] studies execution-based verification of candidate patches. SWE-agent [2] and Agentless [3] motivate structured code access for agents, and RepoGraph [10] repository-level code graphs; the generator here reproduces a specific official schema rather than proposing a new one. The anisotropy of learned embeddings and the effect of removing the common mean are known from word representations [14, 15]; the contribution here is the measurement on code-graph embeddings and its practical consequence. tree-sitter [20] provides the parser.

## References

1. Jimenez, C. E. et al. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024. arXiv:2310.06770.
2. Yang, J. et al. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. NeurIPS 2024. arXiv:2405.15793.
3. Xia, C. S. et al. (2024). Agentless: Demystifying LLM-based Software Engineering Agents. arXiv:2407.01489.
5. Pan, J. et al. (2024). Training Software Engineering Agents and Verifiers with SWE-Gym. arXiv:2412.21139.
6. Yang, J. et al. (2025). SWE-smith: Scaling Data for Software Engineering Agents. arXiv:2504.21798.
7. Badertdinov, I. et al. (2025). SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents. arXiv:2505.20411.
8. Chowdhury, N. et al. (2024). Introducing SWE-bench Verified. OpenAI, August 2024.
10. Ouyang, S. et al. (2024). RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph. arXiv:2410.14684.
14. Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations? EMNLP-IJCNLP 2019. arXiv:1909.00512.
15. Mu, J., Bhat, S., Viswanath, P. (2018). All-but-the-Top: Simple and Effective Postprocessing for Word Representations. ICLR 2018. arXiv:1702.01417.
16. Google (2026). Gemma 4 model card and the Gemma 4 Developer Agent competition pages (data, evaluation, harness rules).
19c. Jain, N. et al. (2025). R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents. arXiv:2504.07164.
20. tree-sitter (Brunsfeld et al.), tree-sitter-python.

**Resources.** Tool and scripts: github.com/debmalyaroy/repograph (Apache-2.0, release v0.1.0 with Linux and Windows binaries). Dataset: kaggle.com/datasets/debmalya84/repograph-verified-swe-tasks ("SWE Tasks"). Environments: kaggle.com/datasets/debmalya84/repograph-verified-swe-tasks-wheels. Notebooks: "Verified SWE tasks: how to use them" and "SWE Tasks Gemma 4 graph guided repair baseline" on Kaggle.
