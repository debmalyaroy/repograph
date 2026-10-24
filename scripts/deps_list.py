# D-053 funnel fix: every requirement a repo's tests may import, from every place projects declare them.
# Runs inside the sandbox image (stdlib only, Python >= 3.11): python3 deps_list.py /ws > reqs.txt
# Sources: [project] dependencies + all optional-dependencies, PEP 735 [dependency-groups] (include-group
# resolved), hatch envs (dependencies, extra-dependencies), poetry dependencies + groups, pdm dev-dependencies,
# uv dev-dependencies, every *requirements*.txt (any depth <= 3, -r includes followed), tox.ini deps.
# Output: one requirement per line, deduplicated, environment markers kept; local paths / VCS / editables dropped.
import re
import sys
import tomllib
from pathlib import Path

GPU_ONLY = re.compile(
    r'^(cudf|cuml|cugraph|cuspatial|cupy|dask-cudf|dask-cuda|rmm|pylibcudf|nvidia-.*|cuda-.*|tensorrt.*|'
    r'jax-cuda.*|jaxlib-cuda.*|onnxruntime-gpu|faiss-gpu.*|nvtx|pynvml|triton|xformers|flash-attn|bitsandbytes)'
    r'(-cu\d+|-cuda\d+x?)?$'
)
SKIP_DIRS = {'.git', 'node_modules', '.tox', '.nox', 'venv', '.venv', 'build', 'dist', 'site-packages'}


def clean(req: str) -> str | None:
    req = req.split(' #', 1)[0].strip()
    if not req or req.startswith(('#', '-e', '--', '.', '/', 'git+', 'http', 'file:')) or '@ ' in req and '://' in req:
        return None
    if req.startswith(('-r ', '-c ')) or '{' in req:  # hatch context templates ({root:uri})
        return None
    name = re.match(r'[A-Za-z0-9][A-Za-z0-9._-]*', req)
    return req if name else None


def poetry_dep(name: str, spec) -> str | None:
    if name.lower() == 'python':
        return None
    if isinstance(spec, dict):
        if 'path' in spec or 'git' in spec or 'url' in spec:
            return None
        spec = spec.get('version', '')
    spec = str(spec).strip()
    if spec in ('', '*'):
        return name
    if spec.startswith(('^', '~')):
        return f'{name}>={spec.lstrip("^~=")}'
    return f'{name}{spec}' if spec[0] in '<>=!~' else f'{name}=={spec}'


def from_pyproject(ws: Path, out: list[str]) -> None:
    p = ws / 'pyproject.toml'
    if not p.exists():
        return
    try:
        d = tomllib.loads(p.read_text(encoding='utf-8', errors='replace'))
    except (tomllib.TOMLDecodeError, UnicodeError):
        return
    proj = d.get('project', {})
    out += proj.get('dependencies', [])
    for deps in proj.get('optional-dependencies', {}).values():
        out += deps
    groups = d.get('dependency-groups', {})
    for deps in groups.values():
        out += [x for x in deps if isinstance(x, str)]  # {include-group = ...} entries resolve to the same groups
    tool = d.get('tool', {})
    for env in (tool.get('hatch', {}).get('envs', {}) or {}).values():
        if isinstance(env, dict):
            out += env.get('dependencies', []) + env.get('extra-dependencies', [])
    poetry = tool.get('poetry', {})
    for name, spec in (poetry.get('dependencies', {}) or {}).items():
        r = poetry_dep(name, spec)
        if r:
            out.append(r)
    for grp in list((poetry.get('group', {}) or {}).values()) + [{'dependencies': poetry.get('dev-dependencies', {})}]:
        for name, spec in (grp.get('dependencies', {}) or {}).items():
            r = poetry_dep(name, spec)
            if r:
                out.append(r)
    for deps in (tool.get('pdm', {}).get('dev-dependencies', {}) or {}).values():
        out += deps
    out += tool.get('uv', {}).get('dev-dependencies', []) or []


def from_requirements(path: Path, out: list[str], seen: set[Path]) -> None:
    if path in seen or not path.exists():
        return
    seen.add(path)
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        m = re.match(r'-r\s+(\S+)|--requirement[ =](\S+)', line)
        if m:
            from_requirements(path.parent / (m.group(1) or m.group(2)), out, seen)
            continue
        out.append(line)


def from_tox(ws: Path, out: list[str]) -> None:
    p = ws / 'tox.ini'
    if not p.exists():
        return
    lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
    in_deps = False
    for line in lines:
        if re.match(r'\s*deps\s*=', line):
            in_deps = True
            rest = line.split('=', 1)[1].strip()
            if rest:
                out.append(rest)
            continue
        if in_deps:
            if line.startswith((' ', '\t')) and line.strip():
                s = line.strip()
                if not s.startswith(('{', '-r')):
                    out.append(re.sub(r'^\S+:\s*', '', s) if re.match(r'^[\w,{}-]+:\s', s) else s)
            else:
                in_deps = False


# pytest plugins implied by test config or fixtures that projects often leave undeclared (verify3 eval round:
# ministack used `worker_id` without pytest-xdist, django-payments set DJANGO_SETTINGS_MODULE without pytest-django).
PLUGIN_HINTS = [
    (r'DJANGO_SETTINGS_MODULE|--ds[= ]|\bdjango_db\b', 'pytest-django'),
    (r'\bworker_id\b|(?:^|\s)-n\s+(?:auto|\d)|--numprocesses|\btestrun_uid\b', 'pytest-xdist'),
    (r'asyncio_mode|pytest\.mark\.asyncio|pytest_asyncio', 'pytest-asyncio'),
    (r'\bmocker\b', 'pytest-mock'),
    (r'--cov\b', 'pytest-cov'),
    (r'^\s*timeout\s*=|pytest\.mark\.timeout', 'pytest-timeout'),
    (r'\bhttpx_mock\b', 'pytest-httpx'),
    (r'\bbenchmark\b\s*[,)]', 'pytest-benchmark'),
    (r'\bfreezer\b', 'pytest-freezer'),
    (r'\baiohttp_client\b', 'pytest-aiohttp'),
    (r'pytest\.mark\.trio\b', 'pytest-trio'),
    (r'^\s*env\s*=', 'pytest-env'),
    (r'\brequests_mock\b', 'requests-mock'),
    (r'pytest\.mark\.flaky|--reruns', 'pytest-rerunfailures'),
    (r'\bsnapshot\b\s*[,)]', 'syrupy'),
    (r'pytest\.mark\.dependency', 'pytest-dependency'),
    (r'pytest\.mark\.order', 'pytest-order'),
    (r'\bdatadir\b', 'pytest-datadir'),
]
# import name -> distribution name where they differ (top-level test imports nobody declared).
IMPORT_DIST = {
    'yaml': 'pyyaml', 'PIL': 'pillow', 'bs4': 'beautifulsoup4', 'dateutil': 'python-dateutil', 'sklearn': 'scikit-learn',
    'cv2': 'opencv-python-headless', 'attr': 'attrs', 'jwt': 'pyjwt', 'dotenv': 'python-dotenv', 'Crypto': 'pycryptodome',
    'google': None, 'gi': None, 'pkg_resources': 'setuptools', 'typing_extensions': 'typing-extensions',
    'magic': 'python-magic', 'serial': 'pyserial', 'zmq': 'pyzmq', 'OpenSSL': 'pyopenssl', 'git': 'gitpython',
    'docx': 'python-docx', 'pptx': 'python-pptx', 'fitz': 'pymupdf', 'multipart': 'python-multipart',
    'hypothesis': 'hypothesis', 'freezegun': 'freezegun', 'faker': 'faker', 'pytz': 'pytz', 'tzdata': 'tzdata',
}
TEST_FILE = re.compile(r'(^|/)(tests?|testing)/|(^|/)(test_[^/]*|[^/]*_test|conftest)\.py$')


def test_hints(ws: Path, out: list[str]) -> None:
    """Plugins implied by pytest config / fixtures, and third-party modules the tests import at top level."""
    cfg_text = ''
    for name in ('pyproject.toml', 'pytest.ini', 'setup.cfg', 'tox.ini'):
        p = ws / name
        if p.exists():
            cfg_text += p.read_text(encoding='utf-8', errors='replace') + '\n'
    local = {p.name.removesuffix('.py') for p in ws.iterdir()} if ws.is_dir() else set()
    for sub in ('src', 'lib'):
        if (ws / sub).is_dir():
            local |= {p.name.removesuffix('.py') for p in (ws / sub).iterdir()}
    test_text, imports = '', set()
    files = [p for p in ws.rglob('*.py') if not (SKIP_DIRS & set(p.parts)) and TEST_FILE.search(p.relative_to(ws).as_posix())]
    for p in files[:400]:
        t = p.read_text(encoding='utf-8', errors='replace')
        test_text += t + '\n'
        for m in re.finditer(r'^(?:import|from)\s+([A-Za-z_]\w*)', t, re.MULTILINE):
            imports.add(m.group(1))
    for pat, dist in PLUGIN_HINTS:
        if re.search(pat, cfg_text, re.MULTILINE) or re.search(pat, test_text, re.MULTILINE):
            out.append(dist)
    stdlib = set(sys.stdlib_module_names) | {'pytest', '_pytest', 'conftest', 'tests', 'test', 'testing', '__future__'}
    for mod in sorted(imports - stdlib - local):
        if mod.startswith('_'):
            continue
        dist = IMPORT_DIST.get(mod, mod.replace('_', '-'))
        if dist:
            out.append(dist)


def norm(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def own_names(ws: Path) -> set[str]:
    """The project's own distribution name(s): pyproject [project] / poetry name, setup.cfg metadata name."""
    names = set()
    p = ws / 'pyproject.toml'
    if p.exists():
        try:
            d = tomllib.loads(p.read_text(encoding='utf-8', errors='replace'))
        except (tomllib.TOMLDecodeError, UnicodeError):
            d = {}
        for n in (d.get('project', {}).get('name'), d.get('tool', {}).get('poetry', {}).get('name')):
            if isinstance(n, str):
                names.add(norm(n))
    cfg = ws / 'setup.cfg'
    if cfg.exists():
        m = re.search(r'^\s*name\s*=\s*(\S+)', cfg.read_text(encoding='utf-8', errors='replace'), re.MULTILINE)
        if m:
            names.add(norm(m.group(1)))
    return names


def main() -> int:
    ws = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    if len(sys.argv) > 2 and sys.argv[2] == '--own':  # wheel-name prefixes of the package under test
        print('\n'.join(n.replace('-', '_') for n in own_names(ws)))
        return 0
    raw: list[str] = []
    from_pyproject(ws, raw)
    seen: set[Path] = set()
    for p in sorted(ws.rglob('*requirements*.txt')):
        if len(p.relative_to(ws).parts) <= 4 and not (SKIP_DIRS & set(p.parts)) and 'docs' not in p.parts:
            from_requirements(p, raw, seen)
    from_tox(ws, raw)
    declared = {norm(m.group(0)) for r in raw if (m := re.match(r'[A-Za-z0-9._-]+', str(r)))}
    hints: list[str] = []
    test_hints(ws, hints)
    raw += [h for h in hints if norm(h) not in declared]  # declared pins win over a bare hinted name
    own = own_names(ws)
    out, names = [], set()
    for r in raw:
        c = clean(str(r))
        if not c or norm(re.match(r'[A-Za-z0-9._-]+', c).group(0)) in own:  # never the package under test
            continue
        if GPU_ONLY.match(norm(re.match(r'[A-Za-z0-9._-]+', c).group(0))):  # CPU sandbox: an installed GPU
            continue  # package makes imports probe CUDA and fail (narwhals: cudf -> libnvrtc not found)
        key = re.sub(r'\s+', '', c).lower()
        if key not in names:
            names.add(key)
            out.append(c)
    out.append('pytest')
    print('\n'.join(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
