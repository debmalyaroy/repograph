//! Build a code graph in the competition schema (NetworkX node-link JSON, `calls` edges) from the
//! Python sources of one repository snapshot.
//!
//! The rules reproduce the official generator, reverse-engineered from the 127 public graphs
//! (doc/V1_PLAN.md §1.5):
//! - node id = module path + qualified name; a leading `*src` directory is dropped, `__init__` too;
//! - `def` and `class` only: `async def` (and everything inside it) is not a node;
//! - nested functions are named under the *current class*, which is a single variable, not a
//!   stack: after a class nested in a method ends, the outer class's later methods lose the prefix;
//! - edges are undirected (stored both ways): class ↔ member, base ↔ subclass, and calls resolved
//!   by bare name or by `Receiver.method` (never `self.x` / `cls.x`);
//! - function text is the source from `def` to the end (no decorators); class text is a summary
//!   (decorators, header, docstring, method stubs with `...`).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use tree_sitter::{Node, Parser};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphNode {
    pub name: String,
    pub text: String,
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphEdge {
    #[serde(rename = "type")]
    pub kind: String,
    pub source: String,
    pub target: String,
    pub key: u32,
}

/// NetworkX node-link JSON, field for field as in `data/graphs/*.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub directed: bool,
    pub multigraph: bool,
    pub graph: BTreeMap<String, String>,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Function,
    Class,
}

struct Def {
    id: String,
    file: usize,
    kind: Kind,
    text: String,
    /// Call references inside the definition: (receiver, name); receiver None = bare call.
    calls: Vec<(Option<String>, String)>,
    /// Base-class names (classes only).
    bases: Vec<String>,
}

/// Module path for a repo-relative file path (forward slashes).
pub fn module_name(path: &str) -> String {
    let stem = path.strip_suffix(".py").unwrap_or(path);
    let mut parts: Vec<&str> = stem
        .split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .collect();
    if parts.len() > 1 && parts[0].ends_with("src") {
        parts.remove(0);
    }
    if parts.last() == Some(&"__init__") {
        parts.pop();
    }
    parts.join(".")
}

fn text<'a>(n: Node, src: &'a [u8]) -> &'a str {
    n.utf8_text(src).unwrap_or("")
}

fn is_async(n: Node) -> bool {
    n.child(0).is_some_and(|c| c.kind() == "async")
}

fn last_name(n: Node, src: &[u8]) -> Option<String> {
    match n.kind() {
        "identifier" => Some(text(n, src).to_string()),
        "attribute" => n
            .child_by_field_name("attribute")
            .map(|a| text(a, src).to_string()),
        "call" | "subscript" => n
            .child_by_field_name("function")
            .or_else(|| n.child_by_field_name("value"))
            .and_then(|f| last_name(f, src)),
        _ => None,
    }
}

/// Calls inside `n` (the whole subtree, nested definitions included, like `ast.walk`).
fn collect_calls(n: Node, src: &[u8], out: &mut Vec<(Option<String>, String)>) {
    if n.kind() == "call"
        && let Some(f) = n.child_by_field_name("function")
    {
        match f.kind() {
            "identifier" => out.push((None, text(f, src).to_string())),
            "attribute" => {
                if let (Some(obj), Some(attr)) = (
                    f.child_by_field_name("object"),
                    f.child_by_field_name("attribute"),
                ) && obj.kind() == "identifier"
                {
                    let recv = text(obj, src);
                    if recv != "self" && recv != "cls" {
                        out.push((Some(recv.to_string()), text(attr, src).to_string()));
                    }
                }
            }
            _ => {}
        }
    }
    let mut cur = n.walk();
    for c in n.children(&mut cur) {
        collect_calls(c, src, out);
    }
}

fn decorators(def: Node, src: &[u8]) -> Vec<String> {
    let Some(p) = def.parent().filter(|p| p.kind() == "decorated_definition") else {
        return vec![];
    };
    let mut cur = p.walk();
    p.children(&mut cur)
        .filter(|c| c.kind() == "decorator")
        .map(|c| text(c, src).trim().to_string())
        .collect()
}

fn docstring(body: Node, src: &[u8]) -> Option<String> {
    let first = body.named_child(0)?;
    if first.kind() != "expression_statement" {
        return None;
    }
    let s = first.named_child(0)?;
    (s.kind() == "string").then(|| text(s, src).to_string())
}

/// `ast.unparse`-style class summary: decorators, header, docstring, method stubs.
fn class_text(n: Node, src: &[u8]) -> String {
    let mut out = String::new();
    for d in decorators(n, src) {
        out.push_str(&d);
        out.push('\n');
    }
    let name = n
        .child_by_field_name("name")
        .map(|x| text(x, src))
        .unwrap_or("");
    let sup = n
        .child_by_field_name("superclasses")
        .map(|x| text(x, src))
        .unwrap_or("");
    out.push_str(&format!("class {name}{sup}:"));
    let mut parts: Vec<String> = vec![];
    if let Some(body) = n.child_by_field_name("body") {
        if let Some(d) = docstring(body, src) {
            parts.push(format!("    {d}"));
        }
        let mut cur = body.walk();
        for c in body.named_children(&mut cur) {
            let f = if c.kind() == "decorated_definition" {
                c.child_by_field_name("definition")
            } else {
                Some(c)
            };
            let Some(f) = f.filter(|f| f.kind() == "function_definition" && !is_async(*f)) else {
                continue;
            };
            let mut stub = String::new();
            for d in decorators(f, src) {
                stub.push_str(&format!("    {d}\n"));
            }
            let fname = f
                .child_by_field_name("name")
                .map(|x| text(x, src))
                .unwrap_or("");
            let params = f
                .child_by_field_name("parameters")
                .map(|x| text(x, src))
                .unwrap_or("()");
            let ret = f
                .child_by_field_name("return_type")
                .map(|x| format!(" -> {}", text(x, src)))
                .unwrap_or_default();
            stub.push_str(&format!("    def {fname}{params}{ret}:\n        ..."));
            parts.push(stub);
        }
    }
    if parts.is_empty() {
        out.push_str("\n    None");
    } else {
        out.push('\n');
        out.push_str(&parts.join("\n\n"));
    }
    out
}

/// Import bindings of one file: local name → qualified dotted path (`from .config import Settings`
/// in `app02/main.py` → `Settings` → `app02.config.Settings`).
fn imports(root: Node, src: &[u8], module: &str, is_pkg: bool) -> HashMap<String, String> {
    let mut out = HashMap::new();
    // package of this module for relative imports
    let mut pkg: Vec<&str> = module.split('.').filter(|p| !p.is_empty()).collect();
    if !is_pkg {
        pkg.pop();
    }
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        match n.kind() {
            "import_statement" => {
                let mut cur = n.walk();
                for c in n.named_children(&mut cur) {
                    match c.kind() {
                        "dotted_name" => {
                            let full = text(c, src).to_string();
                            let head = full.split('.').next().unwrap_or("").to_string();
                            out.insert(head.clone(), head);
                        }
                        "aliased_import" => {
                            if let (Some(name), Some(alias)) = (
                                c.child_by_field_name("name"),
                                c.child_by_field_name("alias"),
                            ) {
                                out.insert(
                                    text(alias, src).to_string(),
                                    text(name, src).to_string(),
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }
            "import_from_statement" => {
                let Some(m) = n.child_by_field_name("module_name") else {
                    continue;
                };
                let base = if m.kind() == "relative_import" {
                    let raw = text(m, src);
                    let dots = raw.chars().take_while(|c| *c == '.').count();
                    let rest = raw[dots..].trim();
                    let keep = pkg.len().saturating_sub(dots.saturating_sub(1));
                    let mut parts: Vec<&str> = pkg[..keep].to_vec();
                    if !rest.is_empty() {
                        parts.extend(rest.split('.'));
                    }
                    parts.join(".")
                } else {
                    text(m, src).to_string()
                };
                let mut cur = n.walk();
                for c in n.children_by_field_name("name", &mut cur) {
                    let (name, alias) = match c.kind() {
                        "aliased_import" => (
                            c.child_by_field_name("name")
                                .map(|x| text(x, src).to_string()),
                            c.child_by_field_name("alias")
                                .map(|x| text(x, src).to_string()),
                        ),
                        _ => (Some(text(c, src).to_string()), None),
                    };
                    if let Some(name) = name {
                        let q = if base.is_empty() {
                            name.clone()
                        } else {
                            format!("{base}.{name}")
                        };
                        out.insert(alias.unwrap_or(name), q);
                    }
                }
            }
            _ => {
                let mut cur = n.walk();
                for c in n.children(&mut cur) {
                    stack.push(c);
                }
            }
        }
    }
    out
}

struct Visitor<'a> {
    src: &'a [u8],
    module: String,
    file: usize,
    cur_class: Option<String>,
    defs: &'a mut Vec<Def>,
}

impl Visitor<'_> {
    fn id(&self, q: &str) -> String {
        if self.module.is_empty() {
            q.to_string()
        } else {
            format!("{}.{q}", self.module)
        }
    }

    fn visit(&mut self, n: Node) {
        match n.kind() {
            "class_definition" => {
                let name = n
                    .child_by_field_name("name")
                    .map(|x| text(x, self.src).to_string())
                    .unwrap_or_default();
                let mut bases = vec![];
                if let Some(sup) = n.child_by_field_name("superclasses") {
                    let mut cur = sup.walk();
                    for b in sup.named_children(&mut cur) {
                        if matches!(b.kind(), "identifier" | "attribute")
                            && let Some(nm) = last_name(b, self.src)
                        {
                            bases.push(nm);
                        }
                    }
                }
                self.defs.push(Def {
                    id: self.id(&name),
                    file: self.file,
                    kind: Kind::Class,
                    text: class_text(n, self.src),
                    calls: vec![],
                    bases,
                });
                self.cur_class = Some(name);
                self.children(n);
                self.cur_class = None;
            }
            "function_definition" => {
                if is_async(n) {
                    return;
                }
                let name = n
                    .child_by_field_name("name")
                    .map(|x| text(x, self.src).to_string())
                    .unwrap_or_default();
                let q = match &self.cur_class {
                    Some(c) => format!("{c}.{name}"),
                    None => name,
                };
                let mut calls = vec![];
                collect_calls(n, self.src, &mut calls);
                self.defs.push(Def {
                    id: self.id(&q),
                    file: self.file,
                    kind: Kind::Function,
                    text: text(n, self.src).to_string(),
                    calls,
                    bases: vec![],
                });
                self.children(n);
            }
            _ => self.children(n),
        }
    }

    fn children(&mut self, n: Node) {
        let mut cur = n.walk();
        let kids: Vec<Node> = n.children(&mut cur).collect();
        for c in kids {
            self.visit(c);
        }
    }
}

/// Build the graph for `files` = (repo-relative path, source). Files that do not parse cleanly are
/// skipped, like `ast.parse` raising `SyntaxError` in the official generator.
pub fn build(repo_name: &str, files: &[(String, String)]) -> anyhow::Result<Graph> {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_python::LANGUAGE.into())?;
    let mut defs: Vec<Def> = vec![];
    let mut file_imports: Vec<HashMap<String, String>> = vec![HashMap::new(); files.len()];
    for (i, (path, src)) in files.iter().enumerate() {
        let Some(tree) = parser.parse(src, None) else {
            continue;
        };
        if tree.root_node().has_error() {
            continue;
        }
        file_imports[i] = imports(
            tree.root_node(),
            src.as_bytes(),
            &module_name(path),
            path.ends_with("__init__.py"),
        );
        let mut v = Visitor {
            src: src.as_bytes(),
            module: module_name(path),
            file: i,
            cur_class: None,
            defs: &mut defs,
        };
        v.visit(tree.root_node());
    }
    // Later definitions of the same id replace earlier ones (dict assignment), keeping first position.
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut uniq: Vec<Def> = vec![];
    for d in defs {
        match index.get(&d.id) {
            Some(&k) => uniq[k] = d,
            None => {
                index.insert(d.id.clone(), uniq.len());
                uniq.push(d);
            }
        }
    }
    let mut short: HashMap<&str, Vec<usize>> = HashMap::new();
    for (k, d) in uniq.iter().enumerate() {
        short
            .entry(d.id.rsplit('.').next().unwrap_or(&d.id))
            .or_default()
            .push(k);
    }
    let parent_of = |id: &str| {
        id.rsplit_once('.')
            .map(|(p, _)| p.to_string())
            .unwrap_or_default()
    };
    let top = |id: &str| parent_of(id).split('.').next().unwrap_or("").to_string();
    let packages: std::collections::HashSet<&str> = uniq
        .iter()
        .map(|d| d.id.split('.').next().unwrap_or(""))
        .collect();
    let resolve = |name: &str, src: usize| -> Vec<usize> {
        let Some(c) = short.get(name) else {
            return vec![];
        };
        if let Some(q) = file_imports[uniq[src].file].get(name) {
            // Imported name: exact definition, else a re-export inside the same repo package;
            // names imported from outside the repository resolve to nothing.
            if let Some(&k) = index.get(q.as_str()) {
                return vec![k];
            }
            let top = q.split('.').next().unwrap_or("");
            if !packages.contains(top) {
                return vec![];
            }
            let m: Vec<usize> = c
                .iter()
                .copied()
                .filter(|&t| uniq[t].id.starts_with(&format!("{top}.")))
                .collect();
            return if m.len() == 1 { m } else { vec![] };
        }
        if c.len() == 1 {
            return c.clone();
        }
        let local: Vec<usize> = c
            .iter()
            .copied()
            .filter(|&t| uniq[t].file == uniq[src].file)
            .collect();
        if !local.is_empty() {
            return local;
        }
        let pkg = top(&uniq[src].id);
        let same: Vec<usize> = c
            .iter()
            .copied()
            .filter(|&t| parent_of(&uniq[t].id).starts_with(&pkg))
            .collect();
        if same.len() <= 3 { same } else { vec![] }
    };
    let mut pairs: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (k, d) in uniq.iter().enumerate() {
        for b in &d.bases {
            for t in resolve(b, k) {
                pairs.insert((t, k));
            }
        }
        if let Some(&p) = index.get(&parent_of(&d.id))
            && uniq[p].kind == Kind::Class
        {
            pairs.insert((p, k));
        }
        if d.kind == Kind::Function {
            for (recv, name) in &d.calls {
                let cands = match recv {
                    None => resolve(name, k),
                    Some(r) => short
                        .get(name.as_str())
                        .map(|c| {
                            c.iter()
                                .copied()
                                .filter(|&t| {
                                    parent_of(&uniq[t].id).rsplit('.').next() == Some(r.as_str())
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                };
                for t in cands {
                    if t != k {
                        pairs.insert((k, t));
                    }
                }
            }
        }
    }
    let sym: BTreeSet<(usize, usize)> = pairs.iter().flat_map(|&(a, b)| [(a, b), (b, a)]).collect();
    let nodes = uniq
        .iter()
        .map(|d| GraphNode {
            name: d.id.clone(),
            text: d.text.clone(),
            id: d.id.clone(),
        })
        .collect();
    let edges = sym
        .into_iter()
        .map(|(a, b)| GraphEdge {
            kind: "calls".into(),
            source: uniq[a].id.clone(),
            target: uniq[b].id.clone(),
            key: 0,
        })
        .collect();
    let mut graph = BTreeMap::new();
    graph.insert("repo_name".to_string(), repo_name.to_string());
    Ok(Graph {
        directed: true,
        multigraph: true,
        graph,
        nodes,
        edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(files: &[(&str, &str)]) -> Graph {
        let f: Vec<(String, String)> = files
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        build("t", &f).unwrap()
    }

    #[test]
    fn module_names() {
        assert_eq!(module_name("pkg/mod.py"), "pkg.mod");
        assert_eq!(module_name("docs_src/a/b.py"), "a.b");
        assert_eq!(module_name("src/pkg/__init__.py"), "pkg");
        assert_eq!(module_name("tests/test_x.py"), "tests.test_x");
    }

    #[test]
    fn single_class_variable_and_async_skip() {
        let src = "class A:\n    def m1(self):\n        class B:\n            def x(self): pass\n    def m2(self): pass\n\nasync def skip():\n    def inner(): pass\n\ndef f():\n    def nested(): pass\n";
        let ids: Vec<String> = g(&[("p/m.py", src)])
            .nodes
            .into_iter()
            .map(|n| n.id)
            .collect();
        assert_eq!(
            ids,
            [
                "p.m.A",
                "p.m.A.m1",
                "p.m.B",
                "p.m.B.x",
                "p.m.m2",
                "p.m.f",
                "p.m.nested"
            ]
        );
    }

    #[test]
    fn edges_member_base_calls_symmetric() {
        let src = "class Base:\n    pass\n\nclass C(Base):\n    def go(self):\n        helper()\n        self.go()\n        Base.make()\n\ndef helper(): pass\n";
        let gr = g(&[("m.py", src)]);
        let e: BTreeSet<(String, String)> = gr
            .edges
            .iter()
            .map(|e| (e.source.clone(), e.target.clone()))
            .collect();
        assert!(
            e.contains(&("m.Base".into(), "m.C".into()))
                && e.contains(&("m.C".into(), "m.Base".into()))
        );
        assert!(e.contains(&("m.C".into(), "m.C.go".into())));
        assert!(e.contains(&("m.C.go".into(), "m.helper".into())));
        assert!(!e.iter().any(|(a, b)| a == "m.C.go" && b == "m.C.go"));
    }

    #[test]
    fn texts() {
        let src = "@dataclass\nclass P:\n    \"\"\"Doc.\"\"\"\n    x: int\n    @property\n    def ok(self) -> bool:\n        return True\n\nclass E:\n    x: int\n";
        let gr = g(&[("m.py", src)]);
        let t: HashMap<String, String> = gr.nodes.into_iter().map(|n| (n.id, n.text)).collect();
        assert_eq!(
            t["m.P"],
            "@dataclass\nclass P:\n    \"\"\"Doc.\"\"\"\n\n    @property\n    def ok(self) -> bool:\n        ..."
        );
        assert_eq!(t["m.P.ok"], "def ok(self) -> bool:\n        return True");
        assert_eq!(t["m.E"], "class E:\n    None");
    }
}
