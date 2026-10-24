//! `repograph mine` (C5/R5): build tasks in the competition format from merged PRs that change
//! source and tests and close an issue. Per PR: base commit = first parent of the merge commit,
//! gold `patch` = non-test changes, `test_patch` = test changes, problem statement = the closing
//! issue (title + body), snapshot = a git repo frozen at the base commit with forward history removed
//! (`snapshots/<instance_id>.tgz`, like the official data), graph = `graphs/<repo>_<commit>.json`.
//! Gold/empty verification and offline wheels are separate steps (they need Docker).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::{graph, source};

#[derive(Debug, Deserialize)]
pub struct PrDetail {
    pub repo: String,
    pub number: u64,
    pub title: Option<String>,
    pub created_at: Option<String>,
    pub merge_commit: Option<String>,
    #[serde(default)]
    pub parents: Vec<String>,
    pub issue: Option<Issue>,
}

#[derive(Debug, Deserialize)]
pub struct Issue {
    pub title: Option<String>,
    pub body: Option<String>,
}

/// One row of `tasks.jsonl`, field for field like the official data.
#[derive(Debug, Serialize)]
pub struct TaskRow {
    pub instance_id: String,
    pub repo: String,
    pub base_commit: String,
    pub problem_statement: String,
    pub hints_text: String,
    pub patch: String,
    pub test_patch: String,
    pub created_at: String,
}

fn git(dir: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .with_context(|| format!("git {args:?}"))?;
    if !out.status.success() {
        bail!(
            "git {args:?} in {}: {}",
            dir.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

pub fn is_test_path(p: &str) -> bool {
    let low = p.to_lowercase();
    let base = low.rsplit('/').next().unwrap_or(&low);
    low.split('/')
        .any(|s| s == "tests" || s == "test" || s == "testing")
        || (base.starts_with("test_") && base.ends_with(".py"))
        || base.ends_with("_test.py")
        || base == "conftest.py"
}

/// Split a unified diff into (source patch, test patch) by file.
pub fn split_diff(diff: &str) -> (String, String) {
    let (mut src, mut tst) = (String::new(), String::new());
    let mut cur: Option<bool> = None;
    for chunk in diff.split_inclusive('\n') {
        if let Some(rest) = chunk.strip_prefix("diff --git a/") {
            let path = rest.split(" b/").next().unwrap_or("");
            cur = Some(is_test_path(path));
        }
        match cur {
            Some(true) => tst.push_str(chunk),
            Some(false) => src.push_str(chunk),
            None => {}
        }
    }
    (src, tst)
}

/// pub129 style (D-053): plain `--- a/` / `+++ b/` hunks, no `diff --git`, `index`, mode or rename
/// lines. Returns None for binary diffs and renames (they cannot be written in that style).
pub fn normalize_diff(diff: &str) -> Option<String> {
    let mut out = String::with_capacity(diff.len());
    for line in diff.split_inclusive('\n') {
        if line.starts_with("GIT binary patch")
            || line.starts_with("Binary files ")
            || line.starts_with("rename from ")
            || line.starts_with("copy from ")
        {
            return None;
        }
        let header = [
            "diff --git ",
            "index ",
            "new file mode ",
            "deleted file mode ",
            "old mode ",
            "new mode ",
            "similarity index ",
            "rename to ",
            "copy to ",
        ];
        if header.iter().any(|h| line.starts_with(h)) {
            continue;
        }
        out.push_str(line);
    }
    Some(out)
}

/// Files a pub129-style or git-style diff creates or deletes (`/dev/null` on one side).
pub fn added_or_deleted(diff: &str) -> Vec<String> {
    let lines: Vec<&str> = diff.lines().collect();
    let mut v = Vec::new();
    for w in lines.windows(2) {
        if w[0] == "--- /dev/null" {
            v.push(w[1].trim_start_matches("+++ b/").to_string());
        } else if w[1] == "+++ /dev/null" {
            v.push(w[0].trim_start_matches("--- a/").to_string());
        }
    }
    v
}

pub fn instance_id(repo: &str, number: u64) -> String {
    let short = repo
        .rsplit('/')
        .next()
        .unwrap_or(repo)
        .to_lowercase()
        .replace(['-', '.'], "_");
    format!("{short}_{number}")
}

/// Ensure a partial clone (all commits, blobs on demand) of `repo` under `clones`.
pub fn ensure_clone(clones: &Path, repo: &str) -> Result<PathBuf> {
    let dir = clones.join(repo.replace('/', "__"));
    if !dir.join(".git").exists() && !dir.join("HEAD").exists() {
        std::fs::create_dir_all(clones)?;
        let status = Command::new("git")
            .args(["clone", "-q", "--filter=blob:none", "--no-checkout"])
            .arg(format!("https://github.com/{repo}.git"))
            .arg(&dir)
            .status()?;
        if !status.success() {
            bail!("clone {repo} failed");
        }
    }
    Ok(dir)
}

/// `out/snapshots/<iid>.tgz`: a clone of `repo` at `base` (50 commits of history, nothing after it,
/// LF working tree, one branch `main`), packed like data/snapshots/*.tgz, plus the graph at base
/// `out/graphs/<repo short>_<base>.json`. Skipped when the snapshot exists; written via `.part`.
pub fn build_snapshot(out: &Path, repo: &str, base: &str, iid: &str) -> Result<()> {
    let snaps = out.join("snapshots");
    std::fs::create_dir_all(&snaps)?;
    let tgz = snaps.join(format!("{iid}.tgz"));
    if tgz.exists() {
        return Ok(());
    }
    let work = out.join("work").join(iid);
    if work.exists() {
        std::fs::remove_dir_all(&work)?;
    }
    std::fs::create_dir_all(&work)?;
    git(&work, &["init", "-q"])?;
    // LF working tree whatever the host's core.autocrlf (PN9: CRLF snapshots broke patches)
    git(&work, &["config", "core.autocrlf", "false"])?;
    git(&work, &["config", "core.eol", "lf"])?;
    let url = format!("https://github.com/{repo}.git");
    git(&work, &["remote", "add", "origin", &url])?;
    git(&work, &["fetch", "-q", "--depth", "50", "origin", base])?;
    git(&work, &["checkout", "-q", "-B", "main", "FETCH_HEAD"])?;
    for r in git(&work, &["for-each-ref", "--format=%(refname)"])?.lines() {
        if r != "refs/heads/main" {
            git(&work, &["update-ref", "-d", r]).ok();
        }
    }
    git(&work, &["remote", "remove", "origin"]).ok();
    git(&work, &["reflog", "expire", "--expire=now", "--all"])?;
    git(&work, &["gc", "-q", "--prune=now"])?;
    let part = snaps.join(format!("{iid}.tgz.part"));
    let st = Command::new("tar")
        .arg("czf")
        .arg(&part)
        .arg("-C")
        .arg(&work)
        .arg(".")
        .status()?;
    if !st.success() {
        bail!("tar failed for {iid}");
    }
    std::fs::rename(&part, &tgz)?;
    let files = source::python_files(&work)?;
    let key = format!(
        "{}_{base}",
        iid.rsplit_once('_').map(|x| x.0).unwrap_or(iid)
    );
    let g = graph::build(&key, &files)?;
    std::fs::create_dir_all(out.join("graphs"))?;
    std::fs::write(
        out.join("graphs").join(format!("{key}.json")),
        serde_json::to_vec(&g)?,
    )?;
    std::fs::remove_dir_all(&work).ok();
    Ok(())
}

/// Build one task; returns None (with a reason) when the PR does not qualify.
pub fn build_task(
    clones: &Path,
    out: &Path,
    pr: &PrDetail,
) -> Result<std::result::Result<TaskRow, String>> {
    let Some(mc) = pr.merge_commit.as_deref() else {
        return Ok(Err("no merge commit".into()));
    };
    let Some(base) = pr.parents.first() else {
        return Ok(Err("no parent".into()));
    };
    let Some(issue) = pr.issue.as_ref() else {
        return Ok(Err("no closing issue".into()));
    };
    let dir = ensure_clone(clones, &pr.repo)?;
    if git(&dir, &["cat-file", "-e", &format!("{mc}^{{commit}}")]).is_err() {
        git(&dir, &["fetch", "-q", "origin", mc]).ok();
    }
    let diff = git(&dir, &["diff", "--no-color", "--no-renames", base, mc])?;
    let (patch, test_patch) = split_diff(&diff);
    if !patch.contains(".py") || test_patch.is_empty() {
        return Ok(Err("needs source .py and test changes".into()));
    }
    let (Some(patch), Some(test_patch)) = (normalize_diff(&patch), normalize_diff(&test_patch))
    else {
        return Ok(Err("binary or renamed file".into()));
    };
    let iid = instance_id(&pr.repo, pr.number);
    build_snapshot(out, &pr.repo, base, &iid)?;
    let problem = format!(
        "{}\n\n{}",
        issue.title.clone().unwrap_or_default(),
        issue.body.clone().unwrap_or_default()
    )
    .trim()
    .to_string();
    Ok(Ok(TaskRow {
        instance_id: iid,
        repo: pr.repo.clone(),
        base_commit: base.clone(),
        problem_statement: problem,
        hints_text: String::new(),
        patch,
        test_patch,
        created_at: pr.created_at.clone().unwrap_or_default(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_source_and_tests() {
        let d = "diff --git a/pkg/m.py b/pkg/m.py\n--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1 +1 @@\n-a\n+b\ndiff --git a/tests/test_m.py b/tests/test_m.py\n--- a/tests/test_m.py\n+++ b/tests/test_m.py\n@@ -1 +1 @@\n-x\n+y\n";
        let (s, t) = split_diff(d);
        assert!(s.contains("pkg/m.py") && !s.contains("tests/"));
        assert!(t.contains("tests/test_m.py") && !t.contains("pkg/m.py"));
        assert_eq!(instance_id("Foo/Bar-Baz.py", 7), "bar_baz_py_7");
        let n = normalize_diff(d).unwrap();
        assert!(!n.contains("diff --git") && n.starts_with("--- a/pkg/m.py\n"));
        let add = "diff --git a/t/test_x.py b/t/test_x.py\nnew file mode 100644\nindex 0000000..e69de29\n--- /dev/null\n+++ b/t/test_x.py\n@@ -0,0 +1 @@\n+x\n";
        let n = normalize_diff(add).unwrap();
        assert_eq!(n, "--- /dev/null\n+++ b/t/test_x.py\n@@ -0,0 +1 @@\n+x\n");
        assert_eq!(added_or_deleted(&n), vec!["t/test_x.py"]);
        assert!(normalize_diff("diff --git a/x b/x\nGIT binary patch\n").is_none());
    }
}
