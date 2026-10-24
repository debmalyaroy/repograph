"""D-053: license of each task's repo, read from its snapshot (no network).

Looks at LICENSE*/COPYING* at the snapshot root, then the `license` field of pyproject.toml / setup.cfg / setup.py,
and classifies the text into an SPDX id. Owner rule: permissive + copyleft ship; unknown stays local.

Usage: uv run python scripts/mining/licenses.py <tasks.jsonl> <snapshot dir> <out.jsonl>
Out rows: {"instance_id", "repo", "license", "class": permissive|copyleft|unknown, "from"}
"""

import json
import re
import sys
import tarfile
from pathlib import Path

# Order matters: the more specific names first.
RULES = [
    ("AGPL-3.0", "copyleft", r"GNU AFFERO GENERAL PUBLIC LICENSE|\bAGPL"),
    ("LGPL-3.0", "copyleft", r"GNU LESSER GENERAL PUBLIC LICENSE\s+Version 3|\bLGPL-?3|LGPLv3"),
    ("LGPL-2.1", "copyleft", r"GNU LESSER GENERAL PUBLIC LICENSE|GNU LIBRARY GENERAL|\bLGPL"),
    ("GPL-3.0", "copyleft", r"GNU GENERAL PUBLIC LICENSE\s+Version 3|\bGPL-?3|GPLv3"),
    ("GPL-2.0", "copyleft", r"GNU GENERAL PUBLIC LICENSE|\bGPL-?2|GPLv2|\bGPL\b"),
    ("MPL-2.0", "copyleft", r"Mozilla Public License|\bMPL"),
    ("EUPL-1.2", "copyleft", r"European Union Public Licen[cs]e|\bEUPL"),
    ("Apache-2.0", "permissive", r"Apache License|\bApache"),
    ("BSD-3-Clause", "permissive", r"Neither the name|BSD[- ]3|3-Clause BSD|New BSD|Modified BSD"),
    (
        "BSD-2-Clause",
        "permissive",
        r"Redistribution and use in source and binary forms|BSD[- ]2|2-Clause BSD|\bBSD\b",
    ),
    ("MIT", "permissive", r"Permission is hereby granted, free of charge|\bMIT\b|\bExpat\b"),
    ("ISC", "permissive", r"Permission to use, copy, modify, and/or distribute|\bISC\b"),
    ("Unlicense", "permissive", r"This is free and unencumbered software|\bUnlicense\b"),
    ("0BSD", "permissive", r"\b0BSD\b|Zero-Clause BSD"),
    ("PSF-2.0", "permissive", r"PYTHON SOFTWARE FOUNDATION LICENSE|\bPSF\b"),
    ("Zlib", "permissive", r"zlib License|\bZlib\b"),
    ("CC0-1.0", "permissive", r"CC0|Creative Commons Zero"),
]
LICENSE_FILE = re.compile(r"^(LICEN[CS]E|COPYING)(\.(md|txt|rst))?$", re.IGNORECASE)
META_FIELD = [
    ("pyproject.toml", re.compile(r'^\s*license\s*=\s*(?:\{\s*text\s*=\s*)?"([^"]+)"', re.MULTILINE)),
    ("pyproject.toml", re.compile(r'"License :: OSI Approved :: ([^"]+)"')),
    ("setup.cfg", re.compile(r"^\s*license\s*=\s*(.+)$", re.MULTILINE)),
    ("setup.py", re.compile(r"license\s*=\s*['\"]([^'\"]+)['\"]")),
]


def classify(text: str):
    for spdx, cls, pat in RULES:
        if re.search(pat, text, re.IGNORECASE if spdx in ("MIT", "ISC", "Zlib", "Unlicense") else 0):
            return spdx, cls
    return None, "unknown"


def scan(tgz: Path):
    """Return (spdx, class, source file) for the snapshot's repo root."""
    texts, metas = [], {}
    with tarfile.open(tgz) as tf:
        names = [m for m in tf.getmembers() if m.isfile()]
        # Snapshots hold the repo at the top level or under one directory.
        depth = min((n.name.strip("./").count("/") for n in names), default=0)
        for m in names:
            rel = m.name.strip("./")
            if rel.count("/") != depth:
                continue
            base = rel.rsplit("/", 1)[-1]
            if LICENSE_FILE.match(base) or base in ("pyproject.toml", "setup.cfg", "setup.py"):
                f = tf.extractfile(m)
                data = f.read(200_000).decode("utf-8", "replace") if f else ""
                if LICENSE_FILE.match(base):
                    texts.append((base, data))
                else:
                    metas[base] = data
    for base, data in texts:
        spdx, cls = classify(data[:6000])
        if spdx:
            return spdx, cls, base
    for fname, pat in META_FIELD:
        m = pat.search(metas.get(fname, ""))
        if m:
            spdx, cls = classify(m.group(1))
            if spdx:
                return spdx, cls, fname
    return None, "unknown", texts[0][0] if texts else None


def main() -> None:
    tasks, snaps, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    counts: dict[str, int] = {}
    with open(out, "w", encoding="utf-8", newline="\n") as w:
        for line in open(tasks, encoding="utf-8"):
            row = json.loads(line)
            iid = row["instance_id"]
            tgz = snaps / f"{iid}.tgz"
            if not tgz.exists():
                print(f"{iid}: no snapshot", file=sys.stderr)
                continue
            try:
                spdx, cls, src = scan(tgz)
            except (tarfile.TarError, OSError) as e:
                print(f"{iid}: {e}", file=sys.stderr)
                spdx, cls, src = None, "unknown", None
            counts[cls] = counts.get(cls, 0) + 1
            w.write(
                json.dumps(
                    {"instance_id": iid, "repo": row.get("repo"), "license": spdx, "class": cls, "from": src}
                )
                + "\n"
            )
    print(f"{tasks}: {counts}")


if __name__ == "__main__":
    main()
