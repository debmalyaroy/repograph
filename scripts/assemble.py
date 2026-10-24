"""D-053 release assembly (X.18): collect the verified tasks of every split into data/release/<split>/.

For each source the sound ids come from verify/by_repo/*.sound (verify3.sh) or a fixed list; tasks, snapshots,
graphs and the repo wheelhouses are HARD-LINKED (same volume, no extra disk). Checks on the way:
  - every split's repos are disjoint from priv_eval's and from pub129's (data/tasks.jsonl)
  - licence from the snapshot (licenses.py): permissive or copyleft ships, unknown is dropped (owner rule)
  - a task needs its snapshot, its graph and a non-empty wheelhouse
Writes <split>/tasks.jsonl (the eight official fields), <split>/meta.jsonl, licenses.jsonl and summary.json.
Run `repograph check` on each split afterwards.

Usage: uv run python scripts/mining/assemble.py [--out data/release] [--dry-run]
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from licenses import scan

FIELDS = [
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "hints_text",
    "patch",
    "test_patch",
    "created_at",
]
PE = Path("data/mine3/priv_eval")
GRAPH_FALLBACK = (Path("data/priv60"), Path("data/priv60r2"))
# split -> [(label, root with snapshots/graphs/wheels_by_repo, tasks.jsonl, ids: 'verify' or a list file)]
SOURCES = {
    "priv_eval": [
        ("priv60", Path("data/priv60"), PE / "priv60.norm.jsonl", PE / "priv61.txt"),
        ("priv60", Path("data/priv60r2"), PE / "priv60r2.norm.jsonl", PE / "priv61.txt"),
        ("priv60-redo", Path("data/mine3/redo60"), Path("data/mine3/redo60/tasks.jsonl"), "verify"),
        ("mine3-eval", Path("data/mine3/eval"), Path("data/mine3/eval/tasks.jsonl"), "verify"),
    ],
    "priv_train": [("mine3-train", Path("data/mine3/train"), Path("data/mine3/train/tasks.jsonl"), "verify")],
    "ext_train": [("ext", Path("data/mine3/ext"), Path("data/mine3/ext/tasks.jsonl"), "verify")],
}


def short(iid: str) -> str:
    return iid.rsplit("_", 1)[0]


def sound_ids(root: Path, ids) -> set[str]:
    if ids != "verify":
        return {line.strip() for line in open(ids, encoding="utf-8") if line.strip()}
    d = root / "verify/by_repo"
    out: set[str] = set()
    for f in d.glob("*.sound") if d.is_dir() else []:
        out |= set(f.read_text(encoding="utf-8").split())
    return out


def link(src: Path, dst: Path, dry: bool) -> None:
    if dry or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, copy_function=os.link)
    else:
        os.link(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/release")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    pub_repos = {json.loads(line)["repo"].lower() for line in open("data/tasks.jsonl", encoding="utf-8")}
    eval_repos: set[str] = set()
    summary, lic_rows = {}, []
    for split, sources in SOURCES.items():
        rows, meta, drops = [], [], Counter()
        seen: set[str] = set()
        for label, root, tasks, ids in sources:
            want = sound_ids(root, ids)
            if not want:
                continue
            for line in open(tasks, encoding="utf-8"):
                t = json.loads(line)
                iid = t["instance_id"]
                if iid not in want or iid in seen:
                    continue
                repo = t["repo"].lower()
                if repo in pub_repos:
                    drops["pub129 repo"] += 1
                    continue
                if split != "priv_eval" and repo in eval_repos:
                    drops["priv_eval repo"] += 1
                    continue
                snap = root / "snapshots" / f"{iid}.tgz"
                gname = f"{short(iid)}_{t['base_commit']}.json"
                # redo60 reuses the graphs built in the priv60 rounds (same base commits)
                graph = next(
                    (
                        r / "graphs" / gname
                        for r in (root, *GRAPH_FALLBACK)
                        if (r / "graphs" / gname).exists()
                    ),
                    root / "graphs" / gname,
                )
                wheels = root / "wheels_by_repo" / short(iid)
                if not snap.exists():
                    drops["no snapshot"] += 1
                    continue
                if not graph.exists():
                    drops["no graph"] += 1
                    continue
                if not wheels.is_dir() or not any(wheels.glob("*.whl")):
                    drops["no wheels"] += 1
                    continue
                spdx, cls, frm = scan(snap)
                if cls == "unknown":
                    drops["no licence"] += 1
                    continue
                seen.add(iid)
                rows.append({k: t.get(k, "") for k in FIELDS})
                meta.append({"instance_id": iid, "source": label, "license": spdx, "license_class": cls})
                lic_rows.append(
                    {
                        "split": split,
                        "instance_id": iid,
                        "repo": t["repo"],
                        "license": spdx,
                        "class": cls,
                        "from": frm,
                    }
                )
                d = out / split
                link(snap, d / "snapshots" / snap.name, a.dry_run)
                link(graph, d / "graphs" / graph.name, a.dry_run)
                link(wheels, d / "wheels_by_repo" / wheels.name, a.dry_run)
        if split == "priv_eval":
            eval_repos = {r["repo"].lower() for r in rows}
        if not a.dry_run and rows:
            (out / split).mkdir(parents=True, exist_ok=True)
            with open(out / split / "tasks.jsonl", "w", encoding="utf-8", newline="\n") as w:
                w.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            with open(out / split / "meta.jsonl", "w", encoding="utf-8", newline="\n") as w:
                w.writelines(json.dumps(m) + "\n" for m in meta)
        summary[split] = {
            "tasks": len(rows),
            "repos": len({r["repo"] for r in rows}),
            "licences": dict(Counter(m["license_class"] for m in meta)),
            "dropped": dict(drops),
        }
        print(split, json.dumps(summary[split]))
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "licenses.jsonl", "w", encoding="utf-8", newline="\n") as w:
            w.writelines(json.dumps(r) + "\n" for r in lic_rows)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
