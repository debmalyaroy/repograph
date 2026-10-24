//! `repograph depcheck`: will a repository's tests run under the official sandbox setup?
//!
//! The scorer's sandbox installs only the project's dependencies, every `optional-dependencies`
//! extra and `requirements*.txt` (plus pytest) — never dependency groups, tox/nox envs or CI
//! files. A task whose tests import (or need as a fixture / pytest plugin) anything else fails
//! for every agent, so the competition's own curation would have dropped it. This check reads
//! the test files and the packaging metadata and lists what is needed but undeclared. It is a
//! pre-filter for mining (V1.C5): transitive dependencies are not resolved, so it can reject a
//! repository whose test import is only pulled in indirectly.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use crate::source;

#[derive(Debug, Serialize)]
pub struct Report {
    pub declared: BTreeSet<String>,
    pub needed: BTreeSet<String>,
    pub undeclared: Vec<String>,
}

/// Distribution names that are always present (pytest and its own dependencies).
const ALWAYS: &[&str] = &[
    "pytest",
    "pluggy",
    "iniconfig",
    "packaging",
    "py",
    "setuptools",
    "pip",
    "wheel",
];

/// Import name → distribution name, where they differ.
const MODULE_DIST: &[(&str, &str)] = &[
    ("yaml", "pyyaml"),
    ("pil", "pillow"),
    ("bs4", "beautifulsoup4"),
    ("sklearn", "scikit-learn"),
    ("skimage", "scikit-image"),
    ("dateutil", "python-dateutil"),
    ("attr", "attrs"),
    ("cv2", "opencv-python"),
    ("jwt", "pyjwt"),
    ("dotenv", "python-dotenv"),
    ("magic", "python-magic"),
    ("crypto", "pycryptodome"),
    ("docx", "python-docx"),
    ("openssl", "pyopenssl"),
    ("git", "gitpython"),
    ("serial", "pyserial"),
    ("usb", "pyusb"),
    ("multipart", "python-multipart"),
    ("google", "protobuf"),
    ("pkg_resources", "setuptools"),
    ("_pytest", "pytest"),
    ("typing_extensions", "typing-extensions"),
    ("zmq", "pyzmq"),
    ("fitz", "pymupdf"),
    ("lxml", "lxml"),
    ("mpl_toolkits", "matplotlib"),
    ("pydantic_core", "pydantic"),
    ("ruamel", "ruamel-yaml"),
    ("slugify", "python-slugify"),
    ("socks", "pysocks"),
    ("websocket", "websocket-client"),
    ("win32api", "pywin32"),
    ("xdist", "pytest-xdist"),
];

/// Fixture → plugin distribution (only unambiguous ones).
const FIXTURE_DIST: &[(&str, &str)] = &[
    ("mocker", "pytest-mock"),
    ("class_mocker", "pytest-mock"),
    ("httpx_mock", "pytest-httpx"),
    ("benchmark", "pytest-benchmark"),
    ("requests_mock", "requests-mock"),
    ("freezer", "pytest-freezer"),
    ("respx_mock", "respx"),
    ("qtbot", "pytest-qt"),
    ("faker", "faker"),
    ("datadir", "pytest-datadir"),
    ("shared_datadir", "pytest-datadir"),
    ("aioresponses", "aioresponses"),
    ("snapshot", "syrupy"),
    ("aiohttp_client", "pytest-aiohttp"),
    ("event_loop", "pytest-asyncio"),
    ("httpserver", "pytest-httpserver"),
    ("fs", "pyfakefs"),
    ("caplog_records", "pytest"),
];

/// Python 3.13 standard library top-level modules (sys.stdlib_module_names, abridged to the
/// importable public ones plus common private ones).
const STDLIB: &[&str] = &[
    "__future__",
    "_thread",
    "abc",
    "argparse",
    "array",
    "ast",
    "asyncio",
    "atexit",
    "base64",
    "bdb",
    "binascii",
    "bisect",
    "builtins",
    "bz2",
    "calendar",
    "cmath",
    "cmd",
    "code",
    "codecs",
    "collections",
    "colorsys",
    "compileall",
    "concurrent",
    "configparser",
    "contextlib",
    "contextvars",
    "copy",
    "copyreg",
    "cProfile",
    "csv",
    "ctypes",
    "curses",
    "dataclasses",
    "datetime",
    "dbm",
    "decimal",
    "difflib",
    "dis",
    "doctest",
    "email",
    "encodings",
    "ensurepip",
    "enum",
    "errno",
    "faulthandler",
    "fcntl",
    "filecmp",
    "fileinput",
    "fnmatch",
    "fractions",
    "ftplib",
    "functools",
    "gc",
    "getopt",
    "getpass",
    "gettext",
    "glob",
    "graphlib",
    "grp",
    "gzip",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "http",
    "imaplib",
    "importlib",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "keyword",
    "linecache",
    "locale",
    "logging",
    "lzma",
    "mailbox",
    "marshal",
    "math",
    "mimetypes",
    "mmap",
    "modulefinder",
    "multiprocessing",
    "netrc",
    "numbers",
    "operator",
    "optparse",
    "os",
    "pathlib",
    "pdb",
    "pickle",
    "pickletools",
    "pkgutil",
    "platform",
    "plistlib",
    "poplib",
    "posix",
    "posixpath",
    "pprint",
    "profile",
    "pstats",
    "pty",
    "pwd",
    "py_compile",
    "pyclbr",
    "pydoc",
    "queue",
    "quopri",
    "random",
    "re",
    "readline",
    "reprlib",
    "resource",
    "rlcompleter",
    "runpy",
    "sched",
    "secrets",
    "select",
    "selectors",
    "shelve",
    "shlex",
    "shutil",
    "signal",
    "site",
    "smtplib",
    "socket",
    "socketserver",
    "sqlite3",
    "ssl",
    "stat",
    "statistics",
    "string",
    "stringprep",
    "struct",
    "subprocess",
    "symtable",
    "sys",
    "sysconfig",
    "syslog",
    "tabnanny",
    "tarfile",
    "tempfile",
    "termios",
    "textwrap",
    "threading",
    "time",
    "timeit",
    "tkinter",
    "token",
    "tokenize",
    "tomllib",
    "trace",
    "traceback",
    "tracemalloc",
    "tty",
    "turtle",
    "types",
    "typing",
    "unicodedata",
    "unittest",
    "urllib",
    "uuid",
    "venv",
    "warnings",
    "wave",
    "weakref",
    "webbrowser",
    "winreg",
    "wsgiref",
    "xml",
    "xmlrpc",
    "zipapp",
    "zipfile",
    "zipimport",
    "zlib",
    "zoneinfo",
    "test",
    "ntpath",
    "nt",
    "msvcrt",
    "winsound",
    "_io",
    "_collections_abc",
    "sre_constants",
    "sre_parse",
    "opcode",
    "genericpath",
    "imp",
    "distutils",
    "lib2to3",
    "asynchat",
    "asyncore",
];

/// PEP 503 normalisation: lowercase, runs of `-_.` → `-`.
pub fn normalize(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut dash = false;
    for c in name.trim().chars() {
        if matches!(c, '-' | '_' | '.') {
            if !dash {
                out.push('-');
            }
            dash = true;
        } else {
            out.extend(c.to_lowercase());
            dash = false;
        }
    }
    out.trim_matches('-').to_string()
}

/// Distribution name of a requirement string (`foo[bar]>=1; python_version<"3.12"` → `foo`).
fn req_name(req: &str) -> Option<String> {
    let s = req.trim();
    if s.is_empty() || s.starts_with('#') || s.starts_with('-') || s.contains("://") {
        return None;
    }
    let end = s
        .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')))
        .unwrap_or(s.len());
    let n = normalize(&s[..end]);
    (!n.is_empty()).then_some(n)
}

fn is_test_path(rel: &str) -> bool {
    let file = rel.rsplit('/').next().unwrap_or(rel);
    rel.ends_with(".py")
        && (rel.starts_with("tests/")
            || rel.starts_with("test/")
            || rel.starts_with("testing/")
            || rel.contains("/tests/")
            || rel.contains("/test/")
            || file == "conftest.py"
            || (file.starts_with("test_") || file.ends_with("_test.py")))
}

fn is_meta_path(rel: &str) -> bool {
    let file = rel.rsplit('/').next().unwrap_or(rel);
    rel == "pyproject.toml"
        || rel == "setup.cfg"
        || rel == "pytest.ini"
        || rel == "tox.ini"
        || (file.starts_with("requirements")
            && file.ends_with(".txt")
            && !rel.contains("/.")
            && rel.matches('/').count() <= 1)
        || (file.starts_with("test-requirements") && file.ends_with(".txt") && !rel.contains('/'))
}

/// A dotted module path: `a`, `a.b_c`.
fn is_module_path(s: &str) -> bool {
    !s.is_empty()
        && s.split('.').all(|seg| {
            !seg.is_empty()
                && !seg.starts_with(|c: char| c.is_ascii_digit())
                && seg.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
}

/// Top-level module names of the unconditional imports in `src`: absolute, at module level (column
/// 0), and syntactically real. Indented imports (inside `try:` / `if` / functions) are optional or
/// lazy and do not make a test fail at collection; docstring prose ("import a module from the
/// docs") does not parse as an import statement.
fn imports(src: &str) -> BTreeSet<String> {
    let top = |m: &str| m.split('.').next().unwrap_or(m).to_string();
    let mut out = BTreeSet::new();
    for line in src.lines() {
        let t = line.split('#').next().unwrap_or("").trim_end();
        if let Some(rest) = t.strip_prefix("from ") {
            let Some((m, _)) = rest.split_once(" import ") else {
                continue;
            };
            if is_module_path(m.trim()) {
                out.insert(top(m.trim()));
            }
        } else if let Some(rest) = t.strip_prefix("import ") {
            let mods: Vec<&str> = rest
                .split(',')
                .map(|p| p.trim().split(" as ").next().unwrap_or("").trim())
                .collect();
            if mods.iter().all(|m| is_module_path(m)) {
                out.extend(mods.iter().map(|m| top(m)));
            }
        }
    }
    out
}

/// Parameter names of `def test_*(...)` and fixtures (single-line signatures are enough here).
fn fixture_params(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for line in src.lines() {
        let t = line.trim_start();
        let t = t.strip_prefix("async ").unwrap_or(t);
        let Some(sig) = t.strip_prefix("def ") else {
            continue;
        };
        let (Some(a), Some(b)) = (sig.find('('), sig.rfind(')')) else {
            continue;
        };
        if b <= a {
            continue;
        }
        for p in sig[a + 1..b].split(',') {
            let p = p.split([':', '=']).next().unwrap_or("").trim();
            if !p.is_empty() && p != "self" && p != "cls" {
                out.insert(p.to_string());
            }
        }
    }
    out
}

/// Distributions declared where the official setup installs from.
fn declared(meta: &[(String, String)]) -> BTreeSet<String> {
    let mut d = BTreeSet::new();
    for (rel, text) in meta {
        if rel == "pyproject.toml" {
            let Ok(v) = text.parse::<toml::Table>() else {
                continue;
            };
            if let Some(p) = v.get("project").and_then(|p| p.as_table()) {
                if let Some(name) = p.get("name").and_then(|n| n.as_str()) {
                    d.insert(normalize(name));
                }
                let reqs = p
                    .get("dependencies")
                    .and_then(|x| x.as_array())
                    .into_iter()
                    .flatten();
                let extras = p
                    .get("optional-dependencies")
                    .and_then(|x| x.as_table())
                    .into_iter()
                    .flat_map(|t| t.values())
                    .filter_map(|x| x.as_array())
                    .flatten();
                for r in reqs.chain(extras).filter_map(|x| x.as_str()) {
                    d.extend(req_name(r));
                }
            }
        } else if rel.ends_with(".txt") {
            d.extend(text.lines().filter_map(req_name));
        }
    }
    d
}

/// Plugins the pytest configuration itself requires (`--cov`, `-n`, `--timeout`, asyncio mode).
fn config_plugins(meta: &[(String, String)]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (rel, text) in meta {
        if !matches!(
            rel.as_str(),
            "pyproject.toml" | "setup.cfg" | "pytest.ini" | "tox.ini"
        ) {
            continue;
        }
        // only the pytest section matters; a crude slice from its header is enough
        let Some(at) = text
            .find("[tool.pytest")
            .or_else(|| text.find("[pytest]"))
            .or_else(|| text.find("[tool:pytest]"))
        else {
            continue;
        };
        let sect = &text[at..];
        let sect = &sect[..sect[1..].find("\n[").map(|i| i + 1).unwrap_or(sect.len())];
        if sect.contains("--cov") {
            out.insert("pytest-cov".into());
        }
        if sect.contains("--timeout") || sect.contains("timeout =") || sect.contains("timeout=") {
            out.insert("pytest-timeout".into());
        }
        if sect.contains(" -n ") || sect.contains("\"-n\"") || sect.contains("'-n'") {
            out.insert("pytest-xdist".into());
        }
        if sect.contains("asyncio_mode") {
            out.insert("pytest-asyncio".into());
        }
    }
    out
}

/// Check a repository directory or snapshot `.tgz`.
pub fn check(root: &Path) -> Result<Report> {
    check_tests(root, None, &[])
}

/// Test files a unified diff touches and the lines it adds to each (`+++ b/<path>` sections).
pub fn diff_added(diff: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = vec![];
    for line in diff.lines() {
        if let Some(p) = line.strip_prefix("+++ ") {
            let p = p.trim().trim_start_matches("b/");
            if p != "/dev/null" {
                out.push((p.to_string(), String::new()));
            }
        } else if let (Some(add), Some(last)) = (line.strip_prefix('+'), out.last_mut()) {
            last.1.push_str(add);
            last.1.push('\n');
        }
    }
    out
}

/// Is `rel` one of `tests`, or a `conftest.py` that pytest loads for one of them (same directory or
/// an ancestor, up to the repository root)?
fn in_scope(rel: &str, tests: &[String]) -> bool {
    if tests.iter().any(|t| t == rel) {
        return true;
    }
    let Some(dir) = rel.strip_suffix("conftest.py") else {
        return false;
    };
    tests.iter().any(|t| t.starts_with(dir))
}

/// As [`check`], but when `tests` is given only those test files (and the conftest.py files above
/// them) count: the scorer runs only the task's test files, so other files' imports do not matter.
/// `extra` adds (path, text) pairs, e.g. the lines a task's test patch adds ([`diff_added`]).
pub fn check_tests(
    root: &Path,
    tests: Option<&[String]>,
    extra: &[(String, String)],
) -> Result<Report> {
    let mut all = source::files(root, |rel| {
        is_test_path(rel) || is_meta_path(rel) || rel.ends_with(".py")
    })?;
    all.extend(extra.iter().cloned());
    let meta: Vec<(String, String)> = all
        .iter()
        .filter(|(r, _)| is_meta_path(r))
        .cloned()
        .collect();
    let declared = declared(&meta);
    // local modules: every directory or file name that holds Python code anywhere in the repo
    let mut local: BTreeSet<String> = BTreeSet::new();
    for (rel, _) in all.iter().filter(|(r, _)| r.ends_with(".py")) {
        for seg in rel.split('/') {
            local.insert(seg.trim_end_matches(".py").to_string());
        }
    }
    let stdlib: BTreeSet<&str> = STDLIB.iter().copied().collect();
    let mut needed = config_plugins(&meta);
    let mut uses_asyncio_mark = false;
    let scoped = |r: &str| match tests {
        Some(t) => in_scope(r, t),
        None => is_test_path(r),
    };
    for (_, src) in all.iter().filter(|(r, _)| scoped(r)) {
        for m in imports(src) {
            if stdlib.contains(m.as_str())
                || local.contains(&m)
                || m.starts_with('_') && m != "_pytest"
            {
                continue;
            }
            let low = m.to_lowercase();
            let dist = MODULE_DIST
                .iter()
                .find(|(k, _)| *k == low)
                .map(|(_, v)| v.to_string())
                .unwrap_or_else(|| normalize(&m));
            needed.insert(dist);
        }
        for p in fixture_params(src) {
            if let Some((_, dist)) = FIXTURE_DIST.iter().find(|(k, _)| *k == p) {
                needed.insert(dist.to_string());
            }
        }
        uses_asyncio_mark |= src.contains("pytest.mark.asyncio");
    }
    if uses_asyncio_mark {
        needed.insert("pytest-asyncio".into());
    }
    let undeclared = needed
        .iter()
        .filter(|n| {
            !ALWAYS.contains(&n.as_str())
                && !declared.contains(*n)
                && !declared.contains(&format!("python-{n}"))
                && !declared.contains(&format!("py{n}"))
        })
        .cloned()
        .collect();
    Ok(Report {
        declared,
        needed,
        undeclared,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requirement_names() {
        assert_eq!(
            req_name("Foo_Bar[x]>=1.0; python_version<'3.12'").as_deref(),
            Some("foo-bar")
        );
        assert_eq!(req_name("-r base.txt"), None);
        assert_eq!(req_name("# comment"), None);
        assert_eq!(normalize("ruamel.yaml"), "ruamel-yaml");
    }

    #[test]
    fn flags_an_undeclared_fixture_plugin_and_import() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(
            p.join("pyproject.toml"),
            "[project]\nname='pkg'\ndependencies=['httpx']\n[project.optional-dependencies]\ntest=['pytest-asyncio']\n\
             [tool.pytest.ini_options]\naddopts='--cov=pkg'\n",
        )
        .unwrap();
        std::fs::create_dir_all(p.join("pkg")).unwrap();
        std::fs::write(p.join("pkg/__init__.py"), "").unwrap();
        std::fs::create_dir_all(p.join("tests")).unwrap();
        std::fs::write(
            p.join("tests/test_a.py"),
            "import os\nimport httpx\nimport yaml\nfrom pkg import x\nfrom tests.helpers import h\n\
             \"\"\"Docs: import a module from the list, from the docs.\"\"\"\n\
             try:\n    import ujson  # optional\nexcept ImportError:\n    ujson = None\n\
             @pytest.mark.asyncio\nasync def test_x(mocker, tmp_path):\n    pass\n",
        )
        .unwrap();
        std::fs::write(p.join("tests/helpers.py"), "").unwrap();
        std::fs::write(p.join("tests/test_other.py"), "import numpy\n").unwrap();
        let r = check(p).unwrap();
        assert_eq!(
            r.undeclared,
            vec!["numpy", "pytest-cov", "pytest-mock", "pyyaml"]
        );
        // scoped to one task's test file: test_other.py's numpy no longer counts; the patch's
        // added lines count even though the file is new
        let added = diff_added("+++ b/tests/test_new.py\n+import attr\n+def test_n(): pass\n");
        let tests = vec![
            "tests/test_a.py".to_string(),
            "tests/test_new.py".to_string(),
        ];
        let r = check_tests(p, Some(&tests), &added).unwrap();
        assert_eq!(
            r.undeclared,
            vec!["attrs", "pytest-cov", "pytest-mock", "pyyaml"]
        );
    }
}
