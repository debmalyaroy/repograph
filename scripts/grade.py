"""Grade patches locally with the OFFICIAL verification (swegemma `verify_task`, Container B) in Docker.

Python because it must call the harness's own verifier (exact fidelity to the scorer's Phase 2).
Normally invoked by `swe grade`; can be run by hand:

    uv run python scripts/grade.py --ids-file ids.txt --patches-dir DIR --out grades.jsonl [--concurrency 2]

A missing `<patches-dir>/<instance_id>.patch` means an empty patch. Output: one JSON line per task
(`instance_id, resolved, test_exit_code, error, duration_s`), appended as each task finishes;
tasks already present in --out are skipped (resumable).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

log = logging.getLogger("grade")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids-file", required=True, type=Path)
    p.add_argument("--patches-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--tasks", type=Path, default=Path("data/tasks.jsonl"))
    p.add_argument("--snapshots", type=Path, default=Path("data/snapshots"))
    p.add_argument("--wheels", type=Path, default=Path("data/wheels"))
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="the graded submission's eval_config timeout_seconds: the scorer runs the hidden tests with "
        "this timeout (verification.py); omitted = harness default 300 s (lenient)",
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        help="keep one persistent harness wheel cache per wheel set under this dir (parallel-safe: "
        "several grade.py processes with different --wheels never share a cache, and a re-grade starts warm)",
    )
    p.add_argument("--image", default="swebench-sandbox:latest")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--no-pin", action="store_true", help="skip per-commit dependency pinning (pure public env)"
    )
    p.add_argument(
        "--sandbox",
        choices=["docker", "subprocess"],
        default="docker",
        help="subprocess = the box grader (no Docker): official verify_task in a clean venv per task, "
        "populated from --template (the sandbox image's site-packages), repo installed editable, pinned",
    )
    p.add_argument(
        "--template", type=Path, help="site-packages dir mirroring the sandbox image (subprocess mode)"
    )
    return p.parse_args()


def _append_line(path: Path, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_test_log(directory: Path, iid: str, text: str) -> None:
    directory.mkdir(exist_ok=True)
    (directory / f"{iid}.log").write_text(text, encoding="utf-8")


# Runs inside the verification container after the harness injected the newest wheel of every
# package (DECISIONS D-023):
# 1. un-shadow the repo: drop any site-packages copy of a package the workspace itself provides
#    (e.g. the rich-15 wheel, pulled in for inline_snapshot, would otherwise replace /workspace/rich);
# 2. re-install each [project] dependency whose injected version violates the range the repo
#    declares at this commit (e.g. old fastapi needs starlette<0.51).
PIN_DEPS = r"""
import glob, json, os, re, shutil, subprocess, sys, tomllib
from importlib import metadata
from pip._vendor.packaging.requirements import Requirement
SP = os.environ.get("PIN_SP", "/usr/local/lib/python3.13/site-packages")
WS = os.environ.get("PIN_WS", "/workspace")
WHEELS = os.environ.get("PIN_WHEELS", "/wheels")
norm = lambda n: re.sub(r"[-_.]+", "_", n).lower()

def editable(d):
    try:
        return json.load(open(os.path.join(d, "direct_url.json"))).get("dir_info", {}).get("editable", False)
    except Exception:
        return False

def remove(name, mods=None):
    for d in glob.glob(f"{SP}/*.dist-info"):
        if norm(os.path.basename(d).split("-")[0]) == norm(name) and not editable(d):
            top = os.path.join(d, "top_level.txt")
            for m in mods or (open(top).read().split() if os.path.exists(top) else [norm(name)]):
                shutil.rmtree(os.path.join(SP, m), ignore_errors=True)
            shutil.rmtree(d, ignore_errors=True)
    for m in mods or []:
        shutil.rmtree(os.path.join(SP, m), ignore_errors=True)

for root in (WS, WS + "/src"):
    for pkg in sorted(glob.glob(f"{root}/*/__init__.py")):
        mod = os.path.basename(os.path.dirname(pkg))
        if mod not in ("tests", "test", "docs") and os.path.isdir(os.path.join(SP, mod)):
            remove(mod, [mod])
            print(f"pin_deps: unshadowed {mod} (site-packages copy removed)")
try:
    deps = tomllib.load(open(WS + "/pyproject.toml", "rb")).get("project", {}).get("dependencies", [])
except Exception as e:
    print("pin_deps: no pyproject deps:", e); sys.exit(0)
for raw in deps:
    try:
        req = Requirement(raw)
    except Exception:
        continue
    if req.marker and not req.marker.evaluate() or not req.specifier:
        continue
    try:
        have = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        continue
    if req.specifier.contains(have, prereleases=True):
        continue
    remove(req.name)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links=" + WHEELS,
                        "--no-deps", "--ignore-installed", f"{req.name}{req.specifier}"], capture_output=True, text=True)
    print(f"pin_deps: {req.name} {have} -> {req.specifier}: {'ok' if r.returncode == 0 else r.stderr.strip()[-200:]}")
"""

# F-045 fallback for boxes that allow no network namespace (no `unshare -n`, no iptables): every Python
# process of the sandbox venv loads this guard through a .pth file, so DNS fails and non-loopback
# connects are refused, as in the official verifier's network_mode='none' container.
OFFLINE_GUARD = r'''
import errno, ipaddress, socket

def _local(host):
    if host in (None, "", "localhost", "localhost.localdomain", "ip6-localhost"):
        return True
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False

_gai = socket.getaddrinfo
def getaddrinfo(host, *a, **kw):
    if isinstance(host, bytes):
        host = host.decode()
    if not _local(host):
        raise socket.gaierror(socket.EAI_NONAME, "offline grading: no DNS")
    return _gai(host, *a, **kw)

def _check(addr):
    if isinstance(addr, tuple) and addr and not _local(addr[0]):
        raise OSError(errno.ENETUNREACH, "offline grading: network is unreachable")

_connect, _connect_ex = socket.socket.connect, socket.socket.connect_ex
def connect(self, addr):
    _check(addr)
    return _connect(self, addr)
def connect_ex(self, addr):
    try:
        _check(addr)
    except OSError as e:
        return e.errno
    return _connect_ex(self, addr)

socket.getaddrinfo = getaddrinfo
socket.socket.connect, socket.socket.connect_ex = connect, connect_ex
'''


def _write_offline_guard(sp: Path) -> None:
    (sp / "_swe_offline.py").write_text(OFFLINE_GUARD, encoding="utf-8")
    (sp / "zz_swe_offline.pth").write_text("import _swe_offline\n", encoding="utf-8")


def _install_pinning(pin_file: Path | None, template: Path | None = None) -> None:
    """Wrap the harness's install_test_dependencies so every verification also runs PIN_DEPS
    (pin_file None = --no-pin: the template fill only, e.g. priv60 repos graded with their own
    per-repo template on the box, D-040).

    Subprocess mode (box grader, D-029/F-026): the harness skips dependency setup for subprocess
    sandboxes and would inherit the grading process's packages. Instead each sandbox venv is created
    empty (system_site_packages=False) and filled here like the Docker image: the template
    site-packages (python:3.13 + the image's pip set + every task wheel), the repo installed editable
    (Docker mode's install_editable_package), then PIN_DEPS with this sandbox's paths."""
    import shutil

    from swegemma.harness import verification

    original = verification.install_test_dependencies
    filled: set[str] = set()

    def wrapped(docker, container_id, *a, **kw):
        original(docker, container_id, *a, **kw)
        env = ""
        if template is not None:
            paths = docker.sandboxes[container_id]
            sp = next(paths["venv"].glob("lib/python*/site-packages"))
            # F-038: the harness can call this twice per sandbox; a second template copy put
            # starlette 1.6's files back over the pinned 0.50 (its dist-info stayed, so PIN_DEPS saw
            # "satisfied") and gold patches of older fastapi commits failed. Fill each venv once.
            if container_id not in filled:
                filled.add(container_id)
                shutil.copytree(template, sp, dirs_exist_ok=True, symlinks=True)
                if os.environ.get("SWE_OFFLINE_GUARD") == "1":
                    _write_offline_guard(sp)
                docker.exec(
                    container_id,
                    "python3 -m pip install -q --no-index --find-links=/wheels --no-build-isolation --no-deps "
                    "-e /workspace 2>/dev/null || true",
                )
            env = f"PIN_SP={sp} PIN_WS={paths['workspace']} PIN_WHEELS=/wheels "
        if pin_file is None:
            return
        docker.copy_to(container_id, pin_file, "/tmp/pin_deps.py")
        res = docker.exec(container_id, env + "python3 /tmp/pin_deps.py")
        if res.stdout.strip():
            log.info("%s %s", container_id[:12], res.stdout.strip().replace("\n", " | "))

    verification.install_test_dependencies = wrapped


def _wheelset_fp(wheels: Path) -> tuple[int, str]:
    import hashlib

    names = sorted(p.name for p in wheels.glob("*.whl"))
    return len(names), hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


def _use_cache_root(root: Path, wheels: Path) -> None:
    """Point the harness's tempfile.gettempdir() (its sp cache lives there) at root/<wheelset fp>."""
    import tempfile

    n, fp = _wheelset_fp(wheels)
    d = (root / fp).resolve()
    d.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(d)
    warm = (d / "swegemma_sp_cache_v8").exists()
    log.info("wheel cache %s (%d wheels, %s) at %s", "warm" if warm else "cold", n, fp, d)


def _invalidate_wheel_cache(wheels: Path) -> None:
    """The harness caches unpacked wheels in <tmp>/swegemma_sp_cache_v8/*.tar and never checks
    whether the wheel set changed. Drop the cache when the wheel file list differs from last run."""
    import hashlib
    import shutil
    import tempfile

    cache = Path(tempfile.gettempdir()) / "swegemma_sp_cache_v8"
    names = sorted(p.name for p in wheels.glob("*.whl"))
    fp = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
    stamp = cache / "wheelset.sha"
    if stamp.exists() and stamp.read_text().strip() == fp:
        log.info("wheel cache valid (%d wheels, %s)", len(names), fp)
        return
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
        log.info("wheel set changed → dropped harness wheel cache %s", cache)
    cache.mkdir(parents=True, exist_ok=True)
    stamp.write_text(fp)
    log.info("wheel cache stamped (%d wheels from %s, %s)", len(names), wheels, fp)


async def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    from adk_submission import ModelRegistry
    from swegemma.config import EvalConfig
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.evaluate import Evaluator
    from swegemma.harness.verification import verify_task
    from swegemma.models import load_tasks

    if args.cache_root:
        _use_cache_root(args.cache_root, args.wheels)
    else:
        await asyncio.to_thread(_invalidate_wheel_cache, args.wheels)
    if not args.no_pin:
        pin_file = args.out.parent / "pin_deps.py"
        pin_file.parent.mkdir(parents=True, exist_ok=True)
        pin_file.write_text(PIN_DEPS, encoding="utf-8")
        _install_pinning(pin_file, args.template if args.sandbox == "subprocess" else None)
    elif args.sandbox == "subprocess":
        _install_pinning(None, args.template)  # the venv still needs the template (was left empty)
    ids = [l.strip() for l in args.ids_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    tasks = {t.instance_id: t for t in load_tasks(args.tasks)}
    done: set[str] = set()
    if args.out.exists():
        # Rows from grader crashes are retried on resume; real verdicts are kept.
        rows = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
        done = {r["instance_id"] for r in rows if not (r.get("error") or "").startswith("grader crash")}
    todo = [i for i in ids if i in tasks and i not in done]
    log.info(
        "grading %d tasks (%d already done, %d unknown) with concurrency %d",
        len(todo),
        len(done & set(ids)),
        len([i for i in ids if i not in tasks]),
        args.concurrency,
    )

    results_dir = args.out.parent / "grade-artifacts"
    # Verification never compiles or calls an agent: submission_dir/models are inert placeholders.
    cfg = EvalConfig(
        submission_dir=Path("data/sample_submission"),
        models=ModelRegistry(),
        tasks_path=args.tasks,
        snapshots_dir=args.snapshots,
        results_dir=results_dir,
        sandbox=args.sandbox,
        image=args.image,
        wheels_dir=args.wheels,
        display_mode="quiet",
        timeout_seconds=args.timeout_seconds,
    )
    evaluator = Evaluator(cfg)
    if args.sandbox == "subprocess":
        if args.template is None or not args.template.is_dir():
            raise SystemExit("--sandbox subprocess needs --template <site-packages dir>")
        evaluator.docker.system_site_packages = False  # nothing inherited from the grading process
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    async def one(iid: str) -> None:
        async with sem:
            task = tasks[iid]
            pf = args.patches_dir / f"{iid}.patch"
            # Bytes, not read_text: universal newlines turned "\r\n" into "\n", so patches of CRLF files
            # (agent diffs and gold alike) no longer applied (F-046). The scorer passes the patch in memory.
            patch = (await asyncio.to_thread(pf.read_bytes)).decode("utf-8") if pf.exists() else ""
            # verify_task measures duration with perf_counter(), so start_time must use it too.
            t0 = time.perf_counter()
            log.info("%s start (patch %d bytes)", iid, len(patch))
            try:
                snap, base, ppath = resolve_task_snapshot_paths(args.snapshots, iid, task.repo)
                res = await verify_task(
                    evaluator.docker,
                    cfg,
                    task,
                    snap,
                    base_snapshot_path=base,
                    patch_path=ppath,
                    agent_patch=patch,
                    start_time=t0,
                )
                row = {
                    "instance_id": iid,
                    "resolved": bool(res.resolved),
                    "test_exit_code": int(res.test_exit_code),
                    "error": res.error,
                    "duration_s": round(time.perf_counter() - t0, 1),
                }
                if res.test_output:
                    await asyncio.to_thread(
                        _write_test_log, args.out.parent / "test_outputs", iid, res.test_output
                    )
            except Exception:  # noqa: BLE001 - one task must never abort the batch; traceback is logged
                tb = traceback.format_exc()
                log.error("%s crashed:\n%s", iid, tb)
                row = {
                    "instance_id": iid,
                    "resolved": False,
                    "test_exit_code": -1,
                    "error": "grader crash: " + tb.splitlines()[-1],
                    "duration_s": round(time.perf_counter() - t0, 1),
                }
            async with lock:
                await asyncio.to_thread(_append_line, args.out, json.dumps(row))
            log.info(
                "%s resolved=%s exit=%s in %.0fs err=%s",
                iid,
                row["resolved"],
                row["test_exit_code"],
                row["duration_s"],
                (row["error"] or "")[:200],
            )

    await asyncio.gather(*(one(i) for i in todo))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
