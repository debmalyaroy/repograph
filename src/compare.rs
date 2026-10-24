//! Parity of a built graph against an official one: node/edge precision and recall, text matches.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::graph::Graph;

#[derive(Debug, Serialize, Default, Clone)]
pub struct Parity {
    pub name: String,
    pub nodes_official: usize,
    pub nodes_built: usize,
    pub node_precision: f64,
    pub node_recall: f64,
    pub edges_official: usize,
    pub edges_built: usize,
    pub edge_precision: f64,
    pub edge_recall: f64,
    /// Exact text match among shared nodes, split by kind (functions start with `def`).
    pub func_text_exact: f64,
    pub class_text_exact: f64,
    pub missing_nodes: Vec<String>,
    pub extra_nodes: Vec<String>,
    pub missing_edges: Vec<(String, String)>,
    pub extra_edges: Vec<(String, String)>,
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 { 1.0 } else { a as f64 / b as f64 }
}

pub fn compare(name: &str, official: &Graph, built: &Graph, samples: usize) -> Parity {
    let on: HashMap<&str, &str> = official
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), n.text.as_str()))
        .collect();
    let bn: HashMap<&str, &str> = built
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), n.text.as_str()))
        .collect();
    let shared: Vec<&str> = on.keys().copied().filter(|k| bn.contains_key(k)).collect();
    let oe: HashSet<(&str, &str)> = official
        .edges
        .iter()
        .map(|e| (e.source.as_str(), e.target.as_str()))
        .collect();
    let be: HashSet<(&str, &str)> = built
        .edges
        .iter()
        .map(|e| (e.source.as_str(), e.target.as_str()))
        .collect();
    let ei = oe.intersection(&be).count();
    let (mut fn_n, mut fn_ok, mut cl_n, mut cl_ok) = (0, 0, 0, 0);
    for k in &shared {
        let o = on[k];
        let is_fn = o.starts_with("def ");
        let ok = o == bn[k];
        if is_fn {
            fn_n += 1;
            fn_ok += ok as usize;
        } else {
            cl_n += 1;
            cl_ok += ok as usize;
        }
    }
    let mut missing_nodes: Vec<String> = on
        .keys()
        .filter(|k| !bn.contains_key(*k))
        .map(|s| s.to_string())
        .collect();
    let mut extra_nodes: Vec<String> = bn
        .keys()
        .filter(|k| !on.contains_key(*k))
        .map(|s| s.to_string())
        .collect();
    let mut missing_edges: Vec<(String, String)> = oe
        .difference(&be)
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    let mut extra_edges: Vec<(String, String)> = be
        .difference(&oe)
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    for v in [&mut missing_nodes, &mut extra_nodes] {
        v.sort();
        v.truncate(samples);
    }
    for v in [&mut missing_edges, &mut extra_edges] {
        v.sort();
        v.truncate(samples);
    }
    Parity {
        name: name.to_string(),
        nodes_official: on.len(),
        nodes_built: bn.len(),
        node_precision: ratio(shared.len(), bn.len()),
        node_recall: ratio(shared.len(), on.len()),
        edges_official: oe.len(),
        edges_built: be.len(),
        edge_precision: ratio(ei, be.len()),
        edge_recall: ratio(ei, oe.len()),
        func_text_exact: ratio(fn_ok, fn_n),
        class_text_exact: ratio(cl_ok, cl_n),
        missing_nodes,
        extra_nodes,
        missing_edges,
        extra_edges,
    }
}
