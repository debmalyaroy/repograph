//! Structural similarity between repositories (C4): a fixed-length fingerprint per code graph,
//! z-scored across the index, compared with cosine.

use std::collections::HashMap;

use serde::Serialize;

use crate::graph::Graph;

/// Fingerprint feature names, in vector order.
pub const FEATURES: &[&str] = &[
    "log_nodes",
    "log_edges",
    "edges_per_node",
    "class_share",
    "method_share",
    "test_share",
    "docs_example_share",
    "mean_depth",
    "max_depth",
    "log_modules",
    "nodes_per_module",
    "mean_text_lines",
    "p90_text_lines",
    "isolated_share",
    "deg_p50",
    "deg_p90",
    "deg_max_log",
    "inherit_share",
    "top_pkg_share",
];

#[derive(Debug, Clone, Serialize)]
pub struct Fingerprint {
    pub name: String,
    pub raw: Vec<f64>,
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn is_test(id: &str) -> bool {
    id.split('.')
        .any(|p| p == "tests" || p == "test" || p.starts_with("test_"))
}

pub fn fingerprint(name: &str, g: &Graph) -> Fingerprint {
    let n = g.nodes.len().max(1) as f64;
    let mut deg: HashMap<&str, usize> = HashMap::new();
    for e in &g.edges {
        *deg.entry(e.source.as_str()).or_default() += 1;
    }
    let ids: Vec<&str> = g.nodes.iter().map(|x| x.id.as_str()).collect();
    let idset: std::collections::HashSet<&str> = ids.iter().copied().collect();
    let classes = g
        .nodes
        .iter()
        .filter(|x| {
            x.text.starts_with("class ") || x.text.starts_with('@') && x.text.contains("\nclass ")
        })
        .count() as f64;
    let methods = ids
        .iter()
        .filter(|id| id.rsplit_once('.').is_some_and(|(p, _)| idset.contains(p)))
        .count() as f64;
    let tests = ids.iter().filter(|id| is_test(id)).count() as f64;
    let docs = ids
        .iter()
        .filter(|id| {
            id.starts_with("docs") || id.starts_with("examples") || id.contains("tutorial")
        })
        .count() as f64;
    let depths: Vec<f64> = ids
        .iter()
        .map(|id| id.matches('.').count() as f64)
        .collect();
    let mut modules: HashMap<&str, usize> = HashMap::new();
    let mut pkgs: HashMap<&str, usize> = HashMap::new();
    for id in &ids {
        let m = id.rsplit_once('.').map(|(p, _)| p).unwrap_or("");
        *modules.entry(m).or_default() += 1;
        *pkgs.entry(id.split('.').next().unwrap_or("")).or_default() += 1;
    }
    let mut lines: Vec<f64> = g
        .nodes
        .iter()
        .map(|x| x.text.lines().count() as f64)
        .collect();
    let mut degs: Vec<f64> = ids
        .iter()
        .map(|id| *deg.get(id).unwrap_or(&0) as f64)
        .collect();
    let isolated = degs.iter().filter(|d| **d == 0.0).count() as f64;
    // inheritance edges: class <-> class pairs that are not parent/child by name
    let class_ids: std::collections::HashSet<&str> = g
        .nodes
        .iter()
        .filter(|x| {
            x.text.starts_with("class ") || x.text.starts_with('@') && x.text.contains("\nclass ")
        })
        .map(|x| x.id.as_str())
        .collect();
    let inherit = g
        .edges
        .iter()
        .filter(|e| class_ids.contains(e.source.as_str()) && class_ids.contains(e.target.as_str()))
        .count() as f64;
    let raw = vec![
        n.ln(),
        (g.edges.len() as f64 + 1.0).ln(),
        g.edges.len() as f64 / n,
        classes / n,
        methods / n,
        tests / n,
        docs / n,
        depths.iter().sum::<f64>() / n,
        depths.iter().cloned().fold(0.0, f64::max),
        (modules.len() as f64 + 1.0).ln(),
        n / modules.len().max(1) as f64,
        lines.iter().sum::<f64>() / n,
        pct(&mut lines, 0.9),
        isolated / n,
        pct(&mut degs.clone(), 0.5),
        pct(&mut degs, 0.9),
        (degs.iter().cloned().fold(0.0, f64::max) + 1.0).ln(),
        inherit / (g.edges.len() as f64 + 1.0),
        pkgs.values().copied().max().unwrap_or(0) as f64 / n,
    ];
    Fingerprint {
        name: name.to_string(),
        raw,
    }
}

/// Z-score every feature across the index; returns standardized vectors in index order.
pub fn standardize(index: &[Fingerprint]) -> Vec<Vec<f64>> {
    let d = FEATURES.len();
    let n = index.len().max(1) as f64;
    let mut mean = vec![0.0; d];
    let mut sd = vec![0.0; d];
    for f in index {
        for (m, x) in mean.iter_mut().zip(&f.raw) {
            *m += x / n;
        }
    }
    for f in index {
        for ((s, m), x) in sd.iter_mut().zip(&mean).zip(&f.raw) {
            *s += (x - m).powi(2) / n;
        }
    }
    index
        .iter()
        .map(|f| {
            (0..d)
                .map(|j| (f.raw[j] - mean[j]) / (sd[j].sqrt() + 1e-9))
                .collect()
        })
        .collect()
}

pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let (mut d, mut x, mut y) = (0.0, 0.0, 0.0);
    for (p, q) in a.iter().zip(b) {
        d += p * q;
        x += p * p;
        y += q * q;
    }
    d / (x.sqrt() * y.sqrt() + 1e-12)
}

/// Rank the index against `query` (an index position); returns (name, similarity), best first.
pub fn rank(index: &[Fingerprint], z: &[Vec<f64>], query: usize, k: usize) -> Vec<(String, f64)> {
    let mut v: Vec<(String, f64)> = (0..index.len())
        .filter(|&i| i != query)
        .map(|i| (index[i].name.clone(), cosine(&z[query], &z[i])))
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(k);
    v
}
