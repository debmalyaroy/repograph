//! Read the Python sources of a repository from a directory or a competition snapshot (`.tgz`).

use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};

/// (repo-relative path with forward slashes, source) for every `.py` file outside `.git/`.
pub fn python_files(root: &Path) -> Result<Vec<(String, String)>> {
    files(root, |rel| rel.ends_with(".py"))
}

/// (repo-relative path, text) for every file outside `.git/` whose path passes `keep`.
pub fn files(root: &Path, keep: impl Fn(&str) -> bool) -> Result<Vec<(String, String)>> {
    if root.is_file() {
        return from_tgz(root, keep);
    }
    let mut out = vec![];
    for e in walkdir::WalkDir::new(root).sort_by_file_name() {
        let e = e?;
        let rel = e
            .path()
            .strip_prefix(root)?
            .to_string_lossy()
            .replace('\\', "/");
        if !e.file_type().is_file()
            || !keep(&rel)
            || rel.starts_with(".git/")
            || rel.contains("/.git/")
        {
            continue;
        }
        let bytes = std::fs::read(e.path()).with_context(|| e.path().display().to_string())?;
        out.push((rel, String::from_utf8_lossy(&bytes).into_owned()));
    }
    Ok(out)
}

fn from_tgz(path: &Path, keep: impl Fn(&str) -> bool) -> Result<Vec<(String, String)>> {
    let f = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    let mut ar = tar::Archive::new(flate2::read::GzDecoder::new(f));
    let mut out = vec![];
    for e in ar.entries()? {
        let mut e = e?;
        if !e.header().entry_type().is_file() {
            continue;
        }
        let name = e.path()?.to_string_lossy().replace('\\', "/");
        let rel = name.trim_start_matches("./").to_string();
        if !keep(&rel) || rel.starts_with(".git/") {
            continue;
        }
        let mut buf = vec![];
        e.read_to_end(&mut buf)?;
        out.push((rel, String::from_utf8_lossy(&buf).into_owned()));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}
