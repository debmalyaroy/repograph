//! Embeddings in the official 256-d space (C2). The official vectors ship as `.npz` archives
//! (`<node_id>.npy`, float32[256]); `read_npz` reads them, `fit_map` learns a ridge-regularised
//! linear map from an open embedding model's space onto them, and `probe` reports fidelity.
//! `struct_features` + `write_npz` make vectors for graphs that have none (F-042: the official
//! space is structural, so hashed name/text tokens propagated over the graph's edges, mapped by a
//! ridge fit on the official vectors, come closest): the harness only offers its graph tools when
//! both a graph and an embedding archive exist for the task.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use nalgebra::{DMatrix, DVector};

/// Read every `<name>.npy` (float32, 1-D) from an `.npz` archive.
pub fn read_npz(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    let f = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    let mut z = zip::ZipArchive::new(f)?;
    let mut out = HashMap::with_capacity(z.len());
    for i in 0..z.len() {
        let mut e = z.by_index(i)?;
        let name = e.name().trim_end_matches(".npy").to_string();
        let mut buf = Vec::with_capacity(e.size() as usize);
        e.read_to_end(&mut buf)?;
        out.insert(
            name,
            parse_npy_f32(&buf).with_context(|| format!("{} in {}", e.name(), path.display()))?,
        );
    }
    Ok(out)
}

fn parse_npy_f32(b: &[u8]) -> Result<Vec<f32>> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        bail!("not an npy file");
    }
    let (hlen, start) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[start..start + hlen])?;
    if !header.contains("'<f4'") {
        bail!("expected little-endian float32, header {header}");
    }
    let data = &b[start + hlen..];
    Ok(data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    d / (na.sqrt() * nb.sqrt() + 1e-12)
}

/// Ridge least squares: W = (XᵀX + λI)⁻¹ XᵀY, with a bias column. X: n×d, Y: n×256.
pub struct LinearMap {
    pub w: DMatrix<f32>,
}

impl LinearMap {
    pub fn fit(x: &[Vec<f32>], y: &[Vec<f32>], lambda: f32) -> Result<Self> {
        let n = x.len();
        let d = x[0].len() + 1;
        let k = y[0].len();
        let xm = DMatrix::from_fn(n, d, |i, j| if j + 1 == d { 1.0 } else { x[i][j] });
        let ym = DMatrix::from_fn(n, k, |i, j| y[i][j]);
        let xt = xm.transpose();
        let a = &xt * &xm + DMatrix::identity(d, d) * lambda;
        let chol = a.cholesky().context("XᵀX + λI not positive definite")?;
        Ok(Self {
            w: chol.solve(&(xt * ym)),
        })
    }

    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut v = DVector::from_column_slice(x).data.as_vec().clone();
        v.push(1.0);
        let row = DMatrix::from_row_slice(1, v.len(), &v);
        (row * &self.w).iter().copied().collect()
    }
}

/// Hashed bag of tokens (node id split on dots, plus the first 2000 chars of its text), log1p, then
/// `hops` rounds of mean over (self + out-neighbours) along the graph's edges. One row per node,
/// in `g.nodes` order.
pub fn struct_features(g: &crate::graph::Graph, dim: usize, hops: usize) -> Vec<Vec<f32>> {
    let pos: HashMap<&str, usize> = g
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.as_str(), i))
        .collect();
    let mut x: Vec<Vec<f32>> = g
        .nodes
        .iter()
        .map(|n| {
            let mut v = vec![0f32; dim];
            let text: String = n.text.chars().take(2000).collect();
            for t in tokens(&n.id.replace('.', " ")).chain(tokens(&text)) {
                v[(fnv1a(t.to_lowercase().as_bytes()) % dim as u64) as usize] += 1.0;
            }
            v.iter_mut().for_each(|e| *e = e.ln_1p());
            v
        })
        .collect();
    let mut nbr: Vec<Vec<usize>> = vec![vec![]; g.nodes.len()];
    for e in &g.edges {
        if let (Some(&a), Some(&b)) = (pos.get(e.source.as_str()), pos.get(e.target.as_str())) {
            nbr[a].push(b);
        }
    }
    for _ in 0..hops {
        let next: Vec<Vec<f32>> = (0..x.len())
            .map(|i| {
                if nbr[i].is_empty() {
                    return x[i].clone();
                }
                let mut v = x[i].clone();
                for &j in &nbr[i] {
                    v.iter_mut().zip(&x[j]).for_each(|(a, b)| *a += b);
                }
                let k = 1.0 + nbr[i].len() as f32;
                v.iter_mut().for_each(|a| *a /= k);
                v
            })
            .collect();
        x = next;
    }
    x
}

/// Identifier-like tokens: CamelCase and snake_case pieces of at least two characters.
fn tokens(s: &str) -> impl Iterator<Item = String> + '_ {
    let mut out = vec![];
    let mut cur = String::new();
    let chars: Vec<char> = s.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        let boundary = !c.is_ascii_alphanumeric()
            || (c.is_ascii_uppercase()
                && i > 0
                && (chars[i - 1].is_ascii_lowercase()
                    || chars.get(i + 1).is_some_and(|n| n.is_ascii_lowercase())
                        && chars[i - 1].is_ascii_uppercase()));
        if boundary && cur.len() >= 2 {
            out.push(std::mem::take(&mut cur));
        } else if boundary {
            cur.clear();
        }
        if c.is_ascii_alphanumeric() {
            cur.push(c);
        }
    }
    if cur.len() >= 2 {
        out.push(cur);
    }
    out.into_iter()
}

fn fnv1a(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, &x| {
        (h ^ x as u64).wrapping_mul(0x100000001b3)
    })
}

/// Write an `.npz` like the official ones: one stored `<name>.npy` (float32, 1-D) per vector.
pub fn write_npz(path: &Path, rows: &[(String, Vec<f32>)]) -> Result<()> {
    let f = std::fs::File::create(path).with_context(|| path.display().to_string())?;
    let mut z = zip::ZipWriter::new(f);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .large_file(false);
    for (name, v) in rows {
        z.start_file(format!("{name}.npy"), opts)?;
        z.write_all(&npy_f32(v))?;
    }
    z.finish()?;
    Ok(())
}

fn npy_f32(v: &[f32]) -> Vec<u8> {
    let mut h = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        v.len()
    );
    // magic (6) + version (2) + header length (2) + header, padded with spaces to a multiple of 64
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend_from_slice(&(h.len() as u16).to_le_bytes());
    b.extend_from_slice(h.as_bytes());
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// Neighbour-rank agreement: for each held-out query, overlap of the top-k cosine neighbours
/// under the official vectors vs the predicted ones (within the held-out set).
pub fn topk_overlap(official: &[Vec<f32>], predicted: &[Vec<f32>], k: usize) -> f64 {
    let n = official.len();
    if n <= k {
        return 0.0;
    }
    let top = |vs: &[Vec<f32>], q: usize| -> Vec<usize> {
        let mut s: Vec<(f32, usize)> = (0..n)
            .filter(|&j| j != q)
            .map(|j| (cosine(&vs[q], &vs[j]), j))
            .collect();
        s.sort_by(|a, b| b.0.total_cmp(&a.0));
        s.into_iter().take(k).map(|(_, j)| j).collect()
    };
    let mut tot = 0usize;
    let queries = n.min(300);
    for q in 0..queries {
        let a = top(official, q);
        let b = top(predicted, q);
        tot += a.iter().filter(|x| b.contains(x)).count();
    }
    tot as f64 / (queries * k) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_roundtrip() {
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        let h = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }          \n";
        b.extend_from_slice(&(h.len() as u16).to_le_bytes());
        b.extend_from_slice(h.as_bytes());
        b.extend_from_slice(&1.5f32.to_le_bytes());
        b.extend_from_slice(&(-2f32).to_le_bytes());
        assert_eq!(parse_npy_f32(&b).unwrap(), vec![1.5, -2.0]);
    }

    #[test]
    fn npz_write_read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.npz");
        let rows = vec![
            ("pkg.mod.Class.method".to_string(), vec![0.25f32, -1.0, 3.0]),
            ("tests.test_a".to_string(), vec![0.0f32; 3]),
        ];
        write_npz(&p, &rows).unwrap();
        let back = read_npz(&p).unwrap();
        assert_eq!(back["pkg.mod.Class.method"], rows[0].1);
        assert_eq!(back.len(), 2);
        assert_eq!(npy_f32(&[1.0]).len() % 64, 4); // header block aligned to 64 bytes
    }

    #[test]
    fn tokens_split_identifiers() {
        let t: Vec<String> = tokens("HTTPConnection parse_header getURLFor x").collect();
        assert_eq!(
            t,
            ["HTTP", "Connection", "parse", "header", "get", "URL", "For"]
        );
    }

    #[test]
    fn linear_map_recovers_a_linear_target() {
        let x: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32 * 0.1, (i % 7) as f32, 1.0 + (i % 3) as f32])
            .collect();
        let y: Vec<Vec<f32>> = x
            .iter()
            .map(|v| vec![2.0 * v[0] - v[1], v[2] + 0.5])
            .collect();
        let m = LinearMap::fit(&x, &y, 1e-4).unwrap();
        let p = m.apply(&x[10]);
        assert!((p[0] - y[10][0]).abs() < 1e-2 && (p[1] - y[10][1]).abs() < 1e-2);
    }
}
