//! `repograph check` (D-053): is a task batch in the competition format, like `data/tasks.jsonl`
//! (pub129)? Per task: exactly the eight official fields, non-empty ids/patches, `patch` touches no
//! test file and `test_patch` only test files, the snapshot `<id>.tgz` exists, is a git checkout at
//! `base_commit` with an LF working tree. Per batch: patch size, file count and issue length
//! quantiles next to the reference set's, so a batch that drifts from pub129's shape shows.

use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;

use crate::mine::is_test_path;

/// The official `tasks.jsonl` fields (data/tasks.jsonl), in order.
pub const FIELDS: [&str; 8] = [
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "hints_text",
    "patch",
    "test_patch",
    "created_at",
];

#[derive(Debug, Serialize)]
pub struct TaskCheck {
    pub instance_id: String,
    pub ok: bool,
    pub problems: Vec<String>,
    pub patch_lines: usize,
    pub patch_files: usize,
    pub statement_chars: usize,
    /// Files the patch or test patch creates or deletes (pub129 has none).
    pub adds_files: bool,
}

/// Files of a unified diff, from its `+++ b/` lines (`--- a/` for deletions); git or plain style.
pub fn diff_files(diff: &str) -> Vec<String> {
    let lines: Vec<&str> = diff.lines().collect();
    let mut v = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if let Some(p) = l.strip_prefix("+++ b/") {
            v.push(p.trim_end().to_string());
        } else if *l == "+++ /dev/null"
            && let Some(p) = i
                .checked_sub(1)
                .and_then(|j| lines[j].strip_prefix("--- a/"))
        {
            v.push(p.trim_end().to_string());
        }
    }
    v
}

pub fn changed_lines(diff: &str) -> usize {
    diff.lines()
        .filter(|l| {
            (l.starts_with('+') || l.starts_with('-'))
                && !l.starts_with("+++")
                && !l.starts_with("---")
        })
        .count()
}

/// HEAD commit and the share of CRLF text among `.py` files of a snapshot.
fn snapshot_facts(tgz: &Path) -> Result<(Option<String>, usize, usize)> {
    let f = std::fs::File::open(tgz)?;
    let mut ar = tar::Archive::new(flate2::read::GzDecoder::new(f));
    let (mut head_ref, mut refs, mut packed) = (None, Vec::new(), String::new());
    let (mut py, mut crlf) = (0, 0);
    for e in ar.entries()? {
        let mut e = e?;
        let name = e
            .path()?
            .to_string_lossy()
            .trim_start_matches("./")
            .to_string();
        if name == ".git/HEAD" {
            let mut s = String::new();
            e.read_to_string(&mut s)?;
            head_ref = Some(s.trim().to_string());
        } else if name.starts_with(".git/refs/heads/") && e.header().entry_type().is_file() {
            let mut s = String::new();
            e.read_to_string(&mut s)?;
            refs.push((
                name.trim_start_matches(".git/").to_string(),
                s.trim().to_string(),
            ));
        } else if name == ".git/packed-refs" {
            e.read_to_string(&mut packed)?;
        } else if name.ends_with(".py") && !name.starts_with(".git/") && py < 200 {
            let mut buf = Vec::new();
            e.read_to_end(&mut buf)?;
            py += 1;
            if buf.windows(2).any(|w| w == b"\r\n") {
                crlf += 1;
            }
        }
    }
    let head = head_ref.and_then(|h| match h.strip_prefix("ref: ") {
        None => Some(h),
        Some(r) => refs
            .iter()
            .find(|(n, _)| n == r)
            .map(|(_, c)| c.clone())
            .or_else(|| {
                packed
                    .lines()
                    .find(|l| l.ends_with(&format!(" {r}")))
                    .map(|l| l.split(' ').next().unwrap_or("").to_string())
            }),
    });
    Ok((head, py, crlf))
}

/// `skip_head`: the official snapshots keep base_commit's tree under rewritten commit ids (pub129:
/// rich_3063 HEAD 306b98bf has the tree of base_commit 368e8ad7), so HEAD is only compared for our own.
pub fn check_task(row: &Value, snapshots: &Path, skip_head: bool) -> TaskCheck {
    let mut problems = Vec::new();
    let obj = row.as_object();
    let keys: BTreeSet<&str> = obj
        .map(|o| o.keys().map(String::as_str).collect())
        .unwrap_or_default();
    let want: BTreeSet<&str> = FIELDS.into_iter().collect();
    if keys != want {
        let extra: Vec<_> = keys.difference(&want).collect();
        let missing: Vec<_> = want.difference(&keys).collect();
        problems.push(format!("fields: extra {extra:?} missing {missing:?}"));
    }
    let s = |k: &str| row.get(k).and_then(Value::as_str).unwrap_or("");
    let iid = s("instance_id").to_string();
    for k in [
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        "patch",
        "test_patch",
    ] {
        if s(k).trim().is_empty() {
            problems.push(format!("empty {k}"));
        }
    }
    if row.get("hints_text").is_some_and(|v| !v.is_string()) {
        problems.push("hints_text not a string".into());
    }
    let (patch, test_patch) = (s("patch"), s("test_patch"));
    let pfiles = diff_files(patch);
    if let Some(t) = pfiles.iter().find(|p| is_test_path(p)) {
        problems.push(format!("patch touches a test file {t}"));
    }
    if !pfiles.iter().any(|p| p.ends_with(".py")) {
        problems.push("patch changes no .py file".into());
    }
    if let Some(t) = diff_files(test_patch).iter().find(|p| !is_test_path(p)) {
        problems.push(format!("test_patch touches a non-test file {t}"));
    }
    if [patch, test_patch].iter().any(|d| {
        d.lines()
            .any(|l| l.starts_with("diff --git ") || l.starts_with("index "))
    }) {
        problems.push("git-style diff headers (pub129 is plain ---/+++)".into());
    }
    let adds_files = [patch, test_patch].iter().any(|d| {
        d.lines()
            .any(|l| l == "--- /dev/null" || l == "+++ /dev/null")
    });
    let tgz = snapshots.join(format!("{iid}.tgz"));
    if !tgz.exists() {
        problems.push("no snapshot".into());
    } else {
        match snapshot_facts(&tgz) {
            Ok((head, py, crlf)) => {
                if head.is_none() || (!skip_head && head.as_deref() != Some(s("base_commit"))) {
                    problems.push(format!("snapshot HEAD {head:?} != base_commit"));
                }
                if py > 0 && crlf * 2 > py {
                    problems.push(format!("CRLF working tree ({crlf}/{py} .py files)"));
                }
            }
            Err(e) => problems.push(format!("snapshot unreadable: {e:#}")),
        }
    }
    TaskCheck {
        ok: problems.is_empty(),
        problems,
        patch_lines: changed_lines(patch),
        patch_files: pfiles.len(),
        statement_chars: s("problem_statement").chars().count(),
        adds_files,
        instance_id: iid,
    }
}

pub fn load_rows(path: &Path) -> Result<Vec<Value>> {
    std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).context("bad JSON line"))
        .collect()
}

/// p25 / p50 / p75 / p90 of a sample.
pub fn quantiles(mut v: Vec<usize>) -> [usize; 4] {
    if v.is_empty() {
        return [0; 4];
    }
    v.sort_unstable();
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    [q(0.25), q(0.5), q(0.75), q(0.9)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_schema_and_test_files() {
        let row = serde_json::json!({
            "instance_id": "x_1", "repo": "o/x", "base_commit": "abc", "problem_statement": "p",
            "hints_text": "", "created_at": "2025-01-01", "FAIL_TO_PASS": [],
            "patch": "diff --git a/tests/test_a.py b/tests/test_a.py\n--- a/tests/test_a.py\n+++ b/tests/test_a.py\n+x\n",
            "test_patch": "--- /dev/null\n+++ b/pkg/a.py\n+y\n"
        });
        let c = check_task(&row, Path::new("/nonexistent"), false);
        assert!(c.adds_files);
        assert!(c.problems.iter().any(|p| p.starts_with("git-style")));
        let all = c.problems.join("; ");
        assert!(
            all.contains("extra") && all.contains("FAIL_TO_PASS"),
            "{all}"
        );
        assert!(all.contains("patch touches a test file"), "{all}");
        assert!(all.contains("non-test file pkg/a.py"), "{all}");
        assert!(all.contains("no snapshot"), "{all}");
        assert_eq!(quantiles(vec![1, 2, 3, 4, 5]), [2, 3, 4, 5]);
    }
}
