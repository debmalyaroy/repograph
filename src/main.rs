//! `repograph` CLI. `build`: graph JSON for a repo dir or snapshot; `parity`: rebuild every official
//! graph from its task snapshot and report node/edge/text agreement (gate C1: nodes ≥ 0.99,
//! edges P/R ≥ 0.90 on all 127).

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use repograph::{check, compare, deps, embed, graph, mine, similar, source};
use tracing::info;

#[derive(Parser)]
#[command(
    name = "repograph",
    version,
    about = "Competition-schema code graphs for Python repositories"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// C5: will the tests run under the official sandbox setup? Lists test imports, fixtures and
    /// pytest plugins that the project does not declare (deps, extras, requirements*.txt).
    /// One JSON line per path (repository directory or snapshot .tgz).
    /// With `--tasks`, checks each task instead: its snapshot plus the lines its test patch adds,
    /// scoped to the task's test files and the conftest.py files above them.
    Depcheck {
        paths: Vec<PathBuf>,
        /// Competition-format tasks.jsonl (instance_id, test_patch).
        #[arg(long)]
        tasks: Option<PathBuf>,
        /// Snapshot dir holding <instance_id>.tgz (with --tasks).
        #[arg(long, default_value = "data/priv60/snapshots")]
        snapshots: PathBuf,
    },
    /// C5/R5: build competition-format tasks (tasks.jsonl, snapshots/, graphs/) from mined PRs.
    Mine {
        /// PR details JSONL (repo, number, merge_commit, parents, issue, created_at).
        #[arg(long, default_value = "data/priv60/mining/pr_details.jsonl")]
        details: PathBuf,
        /// Only these repos (file with one owner/name per line); default: all.
        #[arg(long)]
        repos: Option<PathBuf>,
        #[arg(long, default_value = "data/priv60")]
        out: PathBuf,
        #[arg(long, default_value = "data/priv60/clones")]
        clones: PathBuf,
        #[arg(long, default_value_t = 5)]
        per_repo: usize,
        #[arg(long, default_value_t = 0)]
        limit: usize,
    },
    /// D-053: is a task batch in the competition format (data/tasks.jsonl)? Exactly the eight official
    /// fields, test-free `patch`, test-only `test_patch`, snapshot at base_commit with an LF tree; prints
    /// the batch's patch/issue quantiles next to the reference set's. One JSON line per failing task.
    Check {
        /// Batch tasks.jsonl.
        tasks: PathBuf,
        /// Snapshot dir holding <instance_id>.tgz (default: <tasks dir>/snapshots).
        #[arg(long)]
        snapshots: Option<PathBuf>,
        /// Reference set for the shape comparison.
        #[arg(long, default_value = "data/tasks.jsonl")]
        reference: PathBuf,
        /// Write the passing ids here.
        #[arg(long)]
        ok_ids: Option<PathBuf>,
        /// Official snapshots rewrite commit ids (same tree): do not compare HEAD with base_commit.
        #[arg(long)]
        skip_head: bool,
    },
    /// D-053: snapshots (+ graphs) for any tasks.jsonl with instance_id / repo / base_commit, built like
    /// `mine` (LF tree, 50 commits of history, nothing after base). Resumable; failures go to
    /// <out>/snapshots_failed.txt.
    Snapshots {
        tasks: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Only these ids (one per line).
        #[arg(long)]
        ids: Option<PathBuf>,
        #[arg(long, default_value_t = 8)]
        threads: usize,
    },
    /// D-053: rewrite a tasks.jsonl into the pub129 form: exactly the eight official fields in their
    /// order, plain `---`/`+++` diffs. Every other field goes to `--meta` (one JSON line per task);
    /// rows with binary or renamed files are dropped (listed on stderr).
    Normalize {
        tasks: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Sidecar for the non-official fields (FAIL_TO_PASS, PASS_TO_PASS, version, ...).
        #[arg(long)]
        meta: Option<PathBuf>,
        /// Value of `source` in the sidecar rows.
        #[arg(long, default_value = "")]
        source: String,
    },
    /// Build graphs for every repository directory under `root` (one JSON per repo) in parallel.
    BuildDir {
        root: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
    },
    /// C4: rank repositories by structural similarity to a target graph (fingerprints z-scored over
    /// all graphs found in the given directories). `--validate` runs gate (a): for each official
    /// graph, are the same repo's other commits its nearest neighbours?
    Similar {
        /// Graph JSON files or directories of them.
        #[arg(long, num_args = 1.., default_values = ["data/graphs"])]
        index: Vec<PathBuf>,
        /// Target: a graph file name stem in the index (e.g. requests_0b40…).
        #[arg(long)]
        target: Option<String>,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long)]
        validate: bool,
        /// Write all fingerprints (raw features) as JSONL.
        #[arg(long)]
        dump: Option<PathBuf>,
    },
    /// C2: which open embedding model maps best onto the official 256-d vectors? Fits a linear map
    /// on the train repos and scores the held-out repo (cosine to official, top-10 neighbour overlap).
    EmbedProbe {
        /// Graph keys `<repo>_<commit>`: all but the last are train, the last is held out.
        #[arg(long, num_args = 2.., required = true)]
        keys: Vec<String>,
        /// fastembed models: jina-code, embeddinggemma, bge-small.
        #[arg(long, num_args = 1.., default_values = ["jina-code", "embeddinggemma", "bge-small"])]
        models: Vec<String>,
        /// What is embedded: text | name | name+text.
        #[arg(long, default_value = "text")]
        input: String,
        /// Max nodes per repo.
        #[arg(long, default_value_t = 1500)]
        per_repo: usize,
        #[arg(long, default_value = "data/graphs")]
        graphs: PathBuf,
        #[arg(long, default_value = "data/embeddings")]
        embeddings: PathBuf,
    },
    /// C2 (D-040): write `.npz` embeddings for graphs that have none (priv60), in the official
    /// 256-d space: structural features (hashed tokens propagated over the edges) mapped by a ridge
    /// fit on the official vectors of `--train` graphs. First reports leave-one-repo-out fidelity.
    Embed {
        /// Graph JSON files or directories of them to embed; writes `<out>/<stem>.npz`.
        #[arg(required = true)]
        targets: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        /// Training graph keys `<repo>_<commit>` (default: the newest-named graph of each repo
        /// prefix in --graphs that has an embedding archive).
        #[arg(long, num_args = 1..)]
        train: Vec<String>,
        #[arg(long, default_value = "data/graphs")]
        graphs: PathBuf,
        #[arg(long, default_value = "data/embeddings")]
        embeddings: PathBuf,
        /// Hashed feature width.
        #[arg(long, default_value_t = 1024)]
        dim: usize,
        /// Propagation rounds over the graph's edges.
        #[arg(long, default_value_t = 5)]
        hops: usize,
        #[arg(long, default_value_t = 10.0)]
        lambda: f32,
        /// Max training nodes per repo (evenly spaced).
        #[arg(long, default_value_t = 2000)]
        per_repo: usize,
    },
    /// Build the graph of a repository directory or snapshot `.tgz`.
    Build {
        repo: PathBuf,
        /// Value of graph.repo_name (default: the input's file stem).
        #[arg(long)]
        name: Option<String>,
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Rebuild the official graphs from task snapshots and compare (one task per commit).
    Parity {
        #[arg(long, default_value = "data/tasks.jsonl")]
        tasks: PathBuf,
        #[arg(long, default_value = "data/snapshots")]
        snapshots: PathBuf,
        #[arg(long, default_value = "data/graphs")]
        graphs: PathBuf,
        /// JSONL report, one line per graph.
        #[arg(long, default_value = ".local/research/graphstudy/parity.jsonl")]
        out: PathBuf,
        /// Only the first N commits (0 = all).
        #[arg(long, default_value_t = 0)]
        limit: usize,
        #[arg(long, default_value_t = 8)]
        samples: usize,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?),
        )
        .init();
    match Cli::parse().cmd {
        Cmd::Depcheck {
            paths,
            tasks: Some(tasks),
            snapshots,
        } => {
            let text =
                std::fs::read_to_string(&tasks).with_context(|| tasks.display().to_string())?;
            let rows: Vec<serde_json::Value> = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
            let _ = paths;
            let out: Vec<String> = rows
                .par_iter()
                .map(|t| {
                    let iid = t["instance_id"].as_str().unwrap_or_default();
                    let added: Vec<(String, String)> =
                        deps::diff_added(t["test_patch"].as_str().unwrap_or(""))
                            .into_iter()
                            .filter(|(p, _)| p.ends_with(".py"))
                            .collect();
                    let tests: Vec<String> = added.iter().map(|(p, _)| p.clone()).collect();
                    let snap = snapshots.join(format!("{iid}.tgz"));
                    match deps::check_tests(&snap, Some(&tests), &added) {
                        Ok(r) => serde_json::json!({
                            "instance_id": iid,
                            "ok": r.undeclared.is_empty(),
                            "undeclared": r.undeclared,
                            "tests": tests,
                        })
                        .to_string(),
                        Err(e) => serde_json::json!({"instance_id": iid, "error": e.to_string()})
                            .to_string(),
                    }
                })
                .collect();
            for r in out {
                println!("{r}");
            }
        }
        Cmd::Depcheck { paths, .. } => {
            let rows: Vec<String> = paths
                .par_iter()
                .map(|p| match deps::check(p) {
                    Ok(r) => serde_json::json!({
                        "path": p.to_string_lossy().replace('\\', "/"),
                        "ok": r.undeclared.is_empty(),
                        "undeclared": r.undeclared,
                        "needed": r.needed.len(),
                        "declared": r.declared.len(),
                    })
                    .to_string(),
                    Err(e) => {
                        serde_json::json!({"path": p.to_string_lossy(), "error": e.to_string()})
                            .to_string()
                    }
                })
                .collect();
            for r in rows {
                println!("{r}");
            }
        }
        Cmd::Check {
            tasks,
            snapshots,
            reference,
            ok_ids,
            skip_head,
        } => check_cmd(
            &tasks,
            snapshots.as_deref(),
            &reference,
            ok_ids.as_deref(),
            skip_head,
        )?,
        Cmd::Normalize {
            tasks,
            out,
            meta,
            source,
        } => normalize_cmd(&tasks, &out, meta.as_deref(), &source)?,
        Cmd::Snapshots {
            tasks,
            out,
            ids,
            threads,
        } => snapshots_cmd(&tasks, &out, ids.as_deref(), threads)?,
        Cmd::Mine {
            details,
            repos,
            out,
            clones,
            per_repo,
            limit,
        } => {
            mine_cmd(&details, repos.as_deref(), &out, &clones, per_repo, limit)?;
        }
        Cmd::BuildDir { root, out } => {
            std::fs::create_dir_all(&out)?;
            let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_dir())
                .collect();
            dirs.sort();
            let done: Vec<(String, usize, usize)> = dirs
                .par_iter()
                .filter_map(|d| {
                    let name = d.file_name()?.to_string_lossy().into_owned();
                    let dst = out.join(format!("{name}.json"));
                    if dst.exists() {
                        return None;
                    }
                    let files = source::python_files(d).ok()?;
                    let g = graph::build(&name, &files).ok()?;
                    std::fs::write(&dst, serde_json::to_vec(&g).ok()?).ok()?;
                    Some((name, g.nodes.len(), g.edges.len()))
                })
                .collect();
            println!("built {} graphs into {}", done.len(), out.display());
        }
        Cmd::Similar {
            index,
            target,
            k,
            validate,
            dump,
        } => {
            similar_cmd(&index, target.as_deref(), k, validate, dump.as_deref())?;
        }
        Cmd::EmbedProbe {
            keys,
            models,
            input,
            per_repo,
            graphs,
            embeddings,
        } => {
            #[cfg(feature = "textembed")]
            embed_probe(&keys, &models, &input, per_repo, &graphs, &embeddings)?;
            #[cfg(not(feature = "textembed"))]
            {
                let _ = (keys, models, input, per_repo, graphs, embeddings);
                anyhow::bail!(
                    "embed-probe needs a build with `--features textembed` (fastembed/onnxruntime)"
                );
            }
        }
        Cmd::Embed {
            targets,
            out,
            train,
            graphs,
            embeddings,
            dim,
            hops,
            lambda,
            per_repo,
        } => {
            embed_struct(
                &targets,
                &out,
                &train,
                &graphs,
                &embeddings,
                (dim, hops, lambda, per_repo),
            )?;
        }
        Cmd::Build { repo, name, out } => {
            let name = name.unwrap_or_else(|| {
                repo.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
            let files = source::python_files(&repo)?;
            let g = graph::build(&name, &files)?;
            std::fs::write(&out, serde_json::to_vec(&g)?)?;
            info!(files = files.len(), nodes = g.nodes.len(), edges = g.edges.len(), out = %out.display(), "graph built");
        }
        Cmd::Parity {
            tasks,
            snapshots,
            graphs,
            out,
            limit,
            samples,
        } => {
            // one task per graph key `<repo_short>_<base_commit>`
            let mut by_key: BTreeMap<String, String> = BTreeMap::new();
            for line in std::fs::read_to_string(&tasks)?
                .lines()
                .filter(|l| !l.trim().is_empty())
            {
                let t: serde_json::Value = serde_json::from_str(line)?;
                let iid = t["instance_id"]
                    .as_str()
                    .context("instance_id")?
                    .to_string();
                // `<repo_short>_<number>`: the repo part may hold underscores (vulnerability_db_220)
                let key = format!(
                    "{}_{}",
                    iid.rsplit_once('_').map(|(r, _)| r).unwrap_or(&iid),
                    t["base_commit"].as_str().unwrap_or("")
                );
                by_key.entry(key).or_insert(iid);
            }
            let mut work: Vec<(String, String)> = by_key.into_iter().collect();
            if limit > 0 {
                work.truncate(limit);
            }
            let reports: Vec<compare::Parity> = work
                .par_iter()
                .map(|(key, iid)| -> Result<compare::Parity> {
                    let official: graph::Graph = serde_json::from_slice(&std::fs::read(
                        graphs.join(format!("{key}.json")),
                    )?)?;
                    let files = source::python_files(&snapshots.join(format!("{iid}.tgz")))?;
                    let built = graph::build(key, &files)?;
                    let p = compare::compare(key, &official, &built, samples);
                    info!(
                        key,
                        nodes_r = p.node_recall,
                        nodes_p = p.node_precision,
                        edges_r = p.edge_recall,
                        edges_p = p.edge_precision,
                        "parity"
                    );
                    Ok(p)
                })
                .collect::<Result<_>>()?;
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let mut body = String::new();
            for r in &reports {
                body.push_str(&serde_json::to_string(r)?);
                body.push('\n');
            }
            std::fs::write(&out, body)?;
            let n = reports.len() as f64;
            let mean = |f: fn(&compare::Parity) -> f64| reports.iter().map(f).sum::<f64>() / n;
            let min = |f: fn(&compare::Parity) -> f64| reports.iter().map(f).fold(1.0, f64::min);
            println!("graphs {}", reports.len());
            println!(
                "node precision mean {:.4} min {:.4}",
                mean(|p| p.node_precision),
                min(|p| p.node_precision)
            );
            println!(
                "node recall    mean {:.4} min {:.4}",
                mean(|p| p.node_recall),
                min(|p| p.node_recall)
            );
            println!(
                "edge precision mean {:.4} min {:.4}",
                mean(|p| p.edge_precision),
                min(|p| p.edge_precision)
            );
            println!(
                "edge recall    mean {:.4} min {:.4}",
                mean(|p| p.edge_recall),
                min(|p| p.edge_recall)
            );
            println!(
                "func text exact mean {:.4}; class text exact mean {:.4}",
                mean(|p| p.func_text_exact),
                mean(|p| p.class_text_exact)
            );
            println!("report: {}", out.display());
        }
    }
    Ok(())
}

#[cfg(feature = "textembed")]
fn model_of(name: &str) -> Result<fastembed::EmbeddingModel> {
    use fastembed::EmbeddingModel as M;
    Ok(match name {
        "jina-code" => M::JinaEmbeddingsV2BaseCode,
        "embeddinggemma" => M::EmbeddingGemma300M,
        "bge-small" => M::BGESmallENV15,
        other => anyhow::bail!("unknown model `{other}`"),
    })
}

/// Training rows of one repo: (structural features, official vectors).
type EmbedRows = (Vec<Vec<f32>>, Vec<Vec<f32>>);

/// (features, official vectors) of up to `per_repo` evenly spaced nodes of one official graph.
fn embed_train_rows(
    key: &str,
    graphs: &std::path::Path,
    embeddings: &std::path::Path,
    dim: usize,
    hops: usize,
    per_repo: usize,
) -> Result<EmbedRows> {
    let g: graph::Graph =
        serde_json::from_slice(&std::fs::read(graphs.join(format!("{key}.json")))?)?;
    let off = embed::read_npz(&embeddings.join(format!("{key}.npz")))?;
    let feats = embed::struct_features(&g, dim, hops);
    let idx: Vec<usize> = (0..g.nodes.len())
        .filter(|&i| {
            off.get(&g.nodes[i].id)
                .is_some_and(|v| v.iter().any(|x| *x != 0.0))
        })
        .collect();
    let step = idx.len().div_ceil(per_repo.max(1)).max(1);
    let (mut x, mut y) = (vec![], vec![]);
    for &i in idx.iter().step_by(step) {
        x.push(feats[i].clone());
        y.push(off[&g.nodes[i].id].clone());
    }
    Ok((x, y))
}

fn embed_struct(
    targets: &[PathBuf],
    out: &std::path::Path,
    train: &[String],
    graphs: &std::path::Path,
    embeddings: &std::path::Path,
    (dim, hops, lambda, per_repo): (usize, usize, f32, usize),
) -> Result<()> {
    let train: Vec<String> = if train.is_empty() {
        // one graph per repo prefix: the last key in name order that has an archive
        let mut by_repo: BTreeMap<String, String> = BTreeMap::new();
        for e in std::fs::read_dir(graphs)? {
            let p = e?.path();
            let Some(key) = p
                .file_name()
                .and_then(|s| s.to_str())
                .and_then(|s| s.strip_suffix(".json"))
            else {
                continue;
            };
            if embeddings.join(format!("{key}.npz")).exists()
                && let Some((repo, _)) = key.rsplit_once('_')
            {
                by_repo.insert(repo.to_string(), key.to_string());
            }
        }
        by_repo.into_values().collect()
    } else {
        train.to_vec()
    };
    anyhow::ensure!(train.len() >= 2, "need at least two training graphs");
    let rows: Vec<EmbedRows> = train
        .par_iter()
        .map(|k| embed_train_rows(k, graphs, embeddings, dim, hops, per_repo))
        .collect::<Result<_>>()?;
    // leave-one-repo-out fidelity, the number that says how close the written vectors are
    for (h, key) in train.iter().enumerate() {
        let (mut x, mut y) = (vec![], vec![]);
        for (i, (xi, yi)) in rows.iter().enumerate() {
            if i != h {
                x.extend(xi.iter().cloned());
                y.extend(yi.iter().cloned());
            }
        }
        let m = embed::LinearMap::fit(&x, &y, lambda)?;
        let pred: Vec<Vec<f32>> = rows[h].0.iter().map(|v| m.apply(v)).collect();
        let cos = pred
            .iter()
            .zip(&rows[h].1)
            .map(|(p, o)| embed::cosine(p, o) as f64)
            .sum::<f64>()
            / pred.len().max(1) as f64;
        let top10 = embed::topk_overlap(&rows[h].1, &pred, 10);
        info!(held_out = %key, nodes = pred.len(), cosine = format!("{cos:.3}"), top10_overlap = format!("{top10:.3}"), "embed fidelity");
    }
    let (mut x, mut y) = (vec![], vec![]);
    for (xi, yi) in &rows {
        x.extend(xi.iter().cloned());
        y.extend(yi.iter().cloned());
    }
    let m = embed::LinearMap::fit(&x, &y, lambda)?;
    let mut files = vec![];
    for t in targets {
        if t.is_dir() {
            for e in std::fs::read_dir(t)? {
                let p = e?.path();
                if p.extension().is_some_and(|x| x == "json") {
                    files.push(p);
                }
            }
        } else {
            files.push(t.clone());
        }
    }
    files.sort();
    std::fs::create_dir_all(out)?;
    let written: usize = files
        .par_iter()
        .map(|p| -> Result<usize> {
            let g: graph::Graph = serde_json::from_slice(&std::fs::read(p)?)
                .with_context(|| p.display().to_string())?;
            let feats = embed::struct_features(&g, dim, hops);
            let vecs: Vec<(String, Vec<f32>)> = g
                .nodes
                .iter()
                .zip(&feats)
                .map(|(n, f)| (n.id.clone(), m.apply(f)))
                .collect();
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("graph");
            embed::write_npz(&out.join(format!("{stem}.npz")), &vecs)?;
            Ok(vecs.len())
        })
        .try_reduce(|| 0, |a, b| Ok(a + b))?;
    info!(graphs = files.len(), vectors = written, train = ?train, out = %out.display(), "embeddings written");
    Ok(())
}

#[cfg(feature = "textembed")]
fn embed_probe(
    keys: &[String],
    models: &[String],
    input: &str,
    per_repo: usize,
    graphs: &std::path::Path,
    embeddings: &std::path::Path,
) -> Result<()> {
    // (key, node id, input text, official vector)
    let mut rows: Vec<(usize, String, String, Vec<f32>)> = vec![];
    for (ki, key) in keys.iter().enumerate() {
        let g: graph::Graph =
            serde_json::from_slice(&std::fs::read(graphs.join(format!("{key}.json")))?)?;
        let off = embed::read_npz(&embeddings.join(format!("{key}.npz")))?;
        let mut n = 0;
        for node in &g.nodes {
            let Some(v) = off.get(&node.id) else { continue };
            if v.iter().all(|x| *x == 0.0) {
                continue;
            }
            let text = match input {
                "name" => node.name.clone(),
                "name+text" => format!("{}\n{}", node.name, node.text),
                _ => node.text.clone(),
            };
            rows.push((ki, node.id.clone(), text, v.clone()));
            n += 1;
            if n >= per_repo {
                break;
            }
        }
        info!(key, nodes = n, official = off.len(), "probe rows");
    }
    // the last repo is held out: at least two repos and rows from the held-out one are required
    anyhow::ensure!(
        keys.len() >= 2,
        "embed-probe needs at least two repos (got {})",
        keys.len()
    );
    let held = keys.len() - 1;
    anyhow::ensure!(
        rows.iter().any(|r| r.0 == held) && rows.iter().any(|r| r.0 != held),
        "embed-probe found no nodes in the held-out repo or in the training repos"
    );
    let (train, test): (Vec<_>, Vec<_>) = rows.iter().partition(|r| r.0 != held);
    let off_test: Vec<Vec<f32>> = test.iter().map(|r| r.3.clone()).collect();
    // baseline: how similar are official vectors to each other (collapse) on the held-out repo
    let mut rnd = 0f32;
    for i in 0..off_test.len().min(500) {
        rnd += embed::cosine(&off_test[i], &off_test[(i * 7 + 13) % off_test.len()]);
    }
    println!(
        "held-out {} nodes; official random-pair cosine {:.3}",
        off_test.len(),
        rnd / off_test.len().min(500) as f32
    );
    for m in models {
        let t0 = std::time::Instant::now();
        let mut te = fastembed::TextEmbedding::try_new(
            fastembed::TextInitOptions::new(model_of(m)?)
                .with_show_download_progress(false)
                .with_cache_dir("data/models/fastembed".into()),
        )?;
        let texts: Vec<&str> = rows.iter().map(|r| r.2.as_str()).collect();
        let vecs = te.embed(texts, Some(32))?;
        let dim = vecs[0].len();
        let x_train: Vec<Vec<f32>> = rows
            .iter()
            .zip(&vecs)
            .filter(|(r, _)| r.0 != held)
            .map(|(_, v)| v.clone())
            .collect();
        let y_train: Vec<Vec<f32>> = train.iter().map(|r| r.3.clone()).collect();
        let x_test: Vec<Vec<f32>> = rows
            .iter()
            .zip(&vecs)
            .filter(|(r, _)| r.0 == held)
            .map(|(_, v)| v.clone())
            .collect();
        // direct identity test on the first 256 dims (Matryoshka truncation)
        let direct = if dim >= 256 {
            x_test
                .iter()
                .zip(&off_test)
                .map(|(a, b)| embed::cosine(&a[..256], b))
                .sum::<f32>()
                / x_test.len() as f32
        } else {
            f32::NAN
        };
        let mut best = (f32::MIN, 0f32, 0f64);
        for lambda in [0.1f32, 1.0, 10.0] {
            let map = embed::LinearMap::fit(&x_train, &y_train, lambda)?;
            let pred: Vec<Vec<f32>> = x_test.iter().map(|v| map.apply(v)).collect();
            let cos = pred
                .iter()
                .zip(&off_test)
                .map(|(a, b)| embed::cosine(a, b))
                .sum::<f32>()
                / pred.len() as f32;
            if cos > best.0 {
                best = (cos, lambda, embed::topk_overlap(&off_test, &pred, 10));
            }
        }
        println!(
            "{m:15} dim {dim:4}  direct-256 cos {direct:.3}  mapped cos {:.3} (lambda {})  top-10 neighbour overlap {:.3}  [{:.0}s]",
            best.0,
            best.1,
            best.2,
            t0.elapsed().as_secs_f32()
        );
    }
    Ok(())
}

fn graph_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = vec![];
    for p in paths {
        if p.is_dir() {
            for e in std::fs::read_dir(p)? {
                let e = e?.path();
                if e.extension().is_some_and(|x| x == "json") {
                    out.push(e);
                }
            }
        } else {
            out.push(p.clone());
        }
    }
    out.sort();
    Ok(out)
}

fn similar_cmd(
    paths: &[PathBuf],
    target: Option<&str>,
    k: usize,
    validate: bool,
    dump: Option<&std::path::Path>,
) -> Result<()> {
    let files = graph_files(paths)?;
    let index: Vec<similar::Fingerprint> = files
        .par_iter()
        .filter_map(|f| {
            let g: graph::Graph = serde_json::from_slice(&std::fs::read(f).ok()?).ok()?;
            let name = f.file_stem()?.to_string_lossy().into_owned();
            Some(similar::fingerprint(&name, &g))
        })
        .collect();
    let z = similar::standardize(&index);
    info!(graphs = index.len(), "fingerprints");
    if let Some(d) = dump {
        let mut body = String::new();
        for f in &index {
            body.push_str(&serde_json::to_string(f)?);
            body.push('\n');
        }
        std::fs::write(d, body)?;
    }
    if validate {
        // gate (a): the nearest neighbour of a commit graph is another commit of the same repo
        // graph stems are `<repo>_<commit sha>`; the repo part may hold underscores
        let repo_of = |n: &str| n.rsplit_once('_').map(|(r, _)| r).unwrap_or(n).to_string();
        let (mut hit1, mut hit5, mut q) = (0usize, 0usize, 0usize);
        for i in 0..index.len() {
            let r = repo_of(&index[i].name);
            if index.iter().filter(|f| repo_of(&f.name) == r).count() < 2 {
                continue;
            }
            q += 1;
            let top = similar::rank(&index, &z, i, 5);
            hit1 += usize::from(top.first().is_some_and(|(n, _)| repo_of(n) == r));
            hit5 += usize::from(top.iter().any(|(n, _)| repo_of(n) == r));
        }
        println!("gate (a) same-repo neighbour: top-1 {hit1}/{q}, top-5 {hit5}/{q}");
    }
    if let Some(t) = target {
        let i = index
            .iter()
            .position(|f| f.name == t)
            .with_context(|| format!("{t} not in the index"))?;
        for (n, s) in similar::rank(&index, &z, i, k) {
            println!("{s:.3}  {n}");
        }
    }
    Ok(())
}

fn snapshots_cmd(
    tasks: &std::path::Path,
    out: &std::path::Path,
    ids: Option<&std::path::Path>,
    threads: usize,
) -> Result<()> {
    let only: Option<std::collections::HashSet<String>> = match ids {
        Some(f) => Some(
            std::fs::read_to_string(f)?
                .split_whitespace()
                .map(str::to_string)
                .collect(),
        ),
        None => None,
    };
    let rows: Vec<(String, String, String)> = check::load_rows(tasks)?
        .into_iter()
        .filter_map(|r| {
            let g = |k: &str| r.get(k).and_then(|v| v.as_str()).map(str::to_string);
            Some((g("instance_id")?, g("repo")?, g("base_commit")?))
        })
        .filter(|(i, _, _)| only.as_ref().is_none_or(|o| o.contains(i)))
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()?;
    let failed: Vec<String> = pool.install(|| {
        rows.par_iter()
            .filter_map(|(iid, repo, base)| {
                let t = std::time::Instant::now();
                match mine::build_snapshot(out, repo, base, iid) {
                    Ok(()) => {
                        info!(id = %iid, secs = t.elapsed().as_secs(), "snapshot");
                        None
                    }
                    Err(e) => {
                        let _ = std::fs::remove_dir_all(out.join("work").join(iid));
                        Some(format!("{iid}\t{e:#}"))
                    }
                }
            })
            .collect()
    });
    std::fs::write(out.join("snapshots_failed.txt"), failed.join("\n"))?;
    eprintln!(
        "snapshots: {} tasks, {} failed -> {}",
        rows.len(),
        failed.len(),
        out.join("snapshots").display()
    );
    Ok(())
}

fn normalize_cmd(
    tasks: &std::path::Path,
    out: &std::path::Path,
    meta: Option<&std::path::Path>,
    source: &str,
) -> Result<()> {
    let (mut body, mut side) = (String::new(), String::new());
    let (mut n, mut dropped) = (0, 0);
    for row in check::load_rows(tasks)? {
        let get = |k: &str| {
            row.get(k)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        let iid = get("instance_id");
        let (Some(patch), Some(test_patch)) = (
            mine::normalize_diff(&get("patch")),
            mine::normalize_diff(&get("test_patch")),
        ) else {
            eprintln!("dropped {iid}: binary or renamed file");
            dropped += 1;
            continue;
        };
        let mut o = serde_json::Map::new();
        for k in check::FIELDS {
            let v = match k {
                "patch" => patch.clone(),
                "test_patch" => test_patch.clone(),
                _ => get(k),
            };
            o.insert(k.to_string(), serde_json::Value::String(v));
        }
        body.push_str(&serde_json::to_string(&o)?);
        body.push('\n');
        let mut m = serde_json::Map::new();
        m.insert("instance_id".into(), iid.into());
        if !source.is_empty() {
            m.insert("source".into(), source.into());
        }
        if let Some(obj) = row.as_object() {
            for (k, v) in obj {
                if !check::FIELDS.contains(&k.as_str()) {
                    m.insert(k.clone(), v.clone());
                }
            }
        }
        side.push_str(&serde_json::to_string(&m)?);
        side.push('\n');
        n += 1;
    }
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, body)?;
    if let Some(p) = meta {
        std::fs::write(p, side)?;
    }
    eprintln!("{n} tasks -> {}, {dropped} dropped", out.display());
    Ok(())
}

fn check_cmd(
    tasks: &std::path::Path,
    snapshots: Option<&std::path::Path>,
    reference: &std::path::Path,
    ok_ids: Option<&std::path::Path>,
    skip_head: bool,
) -> Result<()> {
    let snaps = snapshots
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| {
            tasks
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .join("snapshots")
        });
    let rows = check::load_rows(tasks)?;
    let checks: Vec<check::TaskCheck> = rows
        .par_iter()
        .map(|r| check::check_task(r, &snaps, skip_head))
        .collect();
    let mut problems: BTreeMap<String, usize> = BTreeMap::new();
    for c in checks.iter().filter(|c| !c.ok) {
        println!("{}", serde_json::to_string(c)?);
        for p in &c.problems {
            let kind = p.split([' ', ':']).take(3).collect::<Vec<_>>().join(" ");
            *problems.entry(kind).or_default() += 1;
        }
    }
    let ok: Vec<&check::TaskCheck> = checks.iter().filter(|c| c.ok).collect();
    if let Some(p) = ok_ids {
        let ids: Vec<&str> = ok.iter().map(|c| c.instance_id.as_str()).collect();
        std::fs::write(p, ids.join("\n") + "\n")?;
    }
    let refs = check::load_rows(reference)?;
    let shape = |lines: Vec<usize>, files: Vec<usize>, chars: Vec<usize>| {
        format!(
            "patch lines p25/50/75/90 {:?}, files {:?}, issue chars {:?}",
            check::quantiles(lines),
            check::quantiles(files),
            check::quantiles(chars)
        )
    };
    let field = |r: &serde_json::Value, k: &str| {
        r.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string()
    };
    eprintln!(
        "{}: {} tasks, {} ok ({} of them add or delete files), {} failing {:?}",
        tasks.display(),
        checks.len(),
        ok.len(),
        ok.iter().filter(|c| c.adds_files).count(),
        checks.len() - ok.len(),
        problems
    );
    eprintln!(
        "  batch     {}",
        shape(
            ok.iter().map(|c| c.patch_lines).collect(),
            ok.iter().map(|c| c.patch_files).collect(),
            ok.iter().map(|c| c.statement_chars).collect()
        )
    );
    eprintln!(
        "  reference {}",
        shape(
            refs.iter()
                .map(|r| check::changed_lines(&field(r, "patch")))
                .collect(),
            refs.iter()
                .map(|r| check::diff_files(&field(r, "patch")).len())
                .collect(),
            refs.iter()
                .map(|r| field(r, "problem_statement").chars().count())
                .collect()
        )
    );
    Ok(())
}

fn mine_cmd(
    details: &std::path::Path,
    repos: Option<&std::path::Path>,
    out: &std::path::Path,
    clones: &std::path::Path,
    per_repo: usize,
    limit: usize,
) -> Result<()> {
    let only: Option<std::collections::HashSet<String>> = match repos {
        Some(f) => Some(
            std::fs::read_to_string(f)?
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect(),
        ),
        None => None,
    };
    let mut by_repo: BTreeMap<String, Vec<mine::PrDetail>> = BTreeMap::new();
    for line in std::fs::read_to_string(details)?
        .lines()
        .filter(|l| !l.trim().is_empty())
    {
        let Ok(d) = serde_json::from_str::<mine::PrDetail>(line) else {
            continue;
        };
        if only.as_ref().is_some_and(|o| !o.contains(&d.repo)) {
            continue;
        }
        by_repo.entry(d.repo.clone()).or_default().push(d);
    }
    let mut work: Vec<(String, Vec<mine::PrDetail>)> = by_repo.into_iter().collect();
    if limit > 0 {
        work.truncate(limit);
    }
    std::fs::create_dir_all(out)?;
    let results: Vec<(String, Vec<mine::TaskRow>, Vec<String>)> = work
        .into_par_iter()
        .map(|(repo, prs)| {
            let mut rows = vec![];
            let mut skipped = vec![];
            for pr in prs {
                if rows.len() >= per_repo {
                    break;
                }
                match mine::build_task(clones, out, &pr) {
                    Ok(Ok(row)) => rows.push(row),
                    Ok(Err(why)) => skipped.push(format!("{}#{}: {why}", pr.repo, pr.number)),
                    Err(e) => skipped.push(format!("{}#{}: error {e:#}", pr.repo, pr.number)),
                }
            }
            info!(repo = %repo, tasks = rows.len(), skipped = skipped.len(), "mined");
            (repo, rows, skipped)
        })
        .collect();
    let mut body = String::new();
    let mut skipped_all = String::new();
    let (mut n, mut k) = (0, 0);
    for (_, rows, skipped) in &results {
        for r in rows {
            body.push_str(&serde_json::to_string(r)?);
            body.push('\n');
            n += 1;
        }
        for s in skipped {
            skipped_all.push_str(s);
            skipped_all.push('\n');
            k += 1;
        }
    }
    std::fs::write(out.join("tasks.jsonl"), body)?;
    std::fs::write(out.join("skipped.txt"), skipped_all)?;
    println!(
        "{n} tasks, {k} skipped -> {}",
        out.join("tasks.jsonl").display()
    );
    Ok(())
}
