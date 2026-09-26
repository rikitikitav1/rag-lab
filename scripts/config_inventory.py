import ast
import re
import sys
from pathlib import Path

import yaml
from preflight_grid import TUNED

ROOT = Path(__file__).resolve().parent.parent
VERDICTS = ROOT / "docs" / "config_inventory.yaml"
TABLE = ROOT / "docs" / "config_inventory.md"
CLASSES = ("setting", "code", "cap", "seed")
# what a settings file or a data file is, beside the config that names it
FILES = {
    "converters/*/settings/*.json": "a converter's settings arm, hashed whole into a run's record",
    "datasets/converter_gold/manifest.yaml": "the converter gold's documents, cuts and score bar",
    "formats/*.yaml": "a format's vocabulary, shared by every source of that format",
    "sources/*.yaml": "a worked-out source: its declaration and the rules only it needs, seeded into the database",
    "prompts/*.txt": "a prompt version, seeded into the database, the active one a row",
}
_ENV_IN_CODE = re.compile(r"""os\.(?:environ(?:\.get)?\(|environ\[|getenv\()\s*["']([A-Z][A-Z0-9_]+)""")
_ENV_IN_COMPOSE = re.compile(r"\$\{([A-Z][A-Z0-9_]+)")
_ENV_IN_EXAMPLE = re.compile(r"^#?\s*([A-Z][A-Z0-9_]+)=", re.M)
# the preflight's grammar is the one directive: a file and nothing after it
_NOT_MEASURED = re.compile(r"^\s*#\s*not measured")
# where the walk reads code: the app, the converter supervisor beside its image, and the scripts that measure
WALKED = ("app", "converters/supervisor.py", "scripts")
# default arguments that are a threshold or a seed: other defaults are sizes a caller passes anyway
_DECIDING_PARAM = re.compile(r"alpha|seed|threshold|floor|quantile|confidence")


# every key of the loaded config down to a value that is not a model or a mapping; a source's tree is one key
def config_keys() -> dict[str, str]:
    import config
    from pydantic import BaseModel

    holders = {name: yaml.safe_load((ROOT / name).read_text()) or {} for name in _config_files()}
    keys: dict[str, str] = {}

    def where(path: list[str]) -> str:
        holder = next(
            (n for n, doc in holders.items() if _holds(doc, path[:2] if path[0] == "llm" else path[:1])), None
        )
        if holder is None:
            return "app/config.py default"
        raw = holders[holder]
        node = raw
        for part in path:
            if isinstance(node, list):
                node = next((n for n in node if isinstance(n, dict) and n.get("name") == part), None)
            elif isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return "app/config.py default"
            if node is None:
                return "app/config.py default"
        return holder

    def walk(value, path):
        if isinstance(value, BaseModel):
            for name in type(value).model_fields:
                walk(getattr(value, name), path + [name])
        # the coverage map is data, one key: its rows are named in `notes`, not judged one by one
        elif isinstance(value, dict) and value and path != ["technologies"]:
            for k, v in value.items():
                walk(v, path + [str(k)])
        elif isinstance(value, list) and value and all(isinstance(v, BaseModel) for v in value):
            for i, v in enumerate(value):
                walk(v, path + [str(getattr(v, "name", i))])
        else:
            keys[".".join(path)] = where(path)

    # the base layout, whatever overlay this process was started with: the table describes one config
    walk(config._load(str(ROOT / "config.yaml")), [])
    return keys


# the config's files as the loader reads them, without an overlay: it only replaces roles
def _config_files() -> list[str]:
    import config

    return [str(Path(f).relative_to(ROOT)) for f in config.loaded_files(with_overlay=False, with_data=False)]


def _directive(comment: str) -> str | None:
    if m := TUNED.match(comment):
        return f"tuned: file={m.group(1)}"
    return "not measured" if _NOT_MEASURED.match(comment) else None


def _holds(doc: dict, path: list[str]) -> bool:
    for part in path:
        if not isinstance(doc, dict) or part not in doc:
            return False
        doc = doc[part]
    return True


# how each config key was chosen: its own directive, or the nearest parent's, as the yaml nests them
def config_annotations() -> dict[str, str]:
    # a file's last directive belongs to no key of the next file
    lines = [line for name in _config_files() for line in [*(ROOT / name).read_text().splitlines(), None]]
    path, bound, pending = [], {}, None
    for line in lines:
        if line is None:
            path, pending = [], None
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            pending = _directive(stripped) or pending
            continue
        m = re.match(r"^(\s*)([A-Za-z_][\w-]*):", line)
        if not m:
            continue
        depth = len(m.group(1)) // 2
        path = path[:depth] + [m.group(2)]
        tail = _directive("#" + line.split(" #", 1)[1].strip()) if " #" in line else None
        said = tail or pending
        if said:
            bound[".".join(path)] = said
        pending = None
    found = {}
    for key, where in config_keys().items():
        # a default the model holds was never read by the measurement its parent names
        if where == "app/config.py default" or where.startswith("datasets/"):
            continue
        parts = key.split(".")
        for n in range(len(parts), 0, -1):
            if ".".join(parts[:n]) in bound:
                found[key] = bound[".".join(parts[:n])]
                break
    return found


def _walked_files():
    for name in WALKED:
        target = ROOT / name
        yield from ([target] if target.is_file() else sorted(target.rglob("*.py")))


def env_sets() -> dict[str, set[str]]:
    import config

    code = set()
    for source in _walked_files():
        code |= set(_ENV_IN_CODE.findall(source.read_text()))
    # an engine's address and key are read by its prefix; a cloud engine's prefix lives in the database only
    code |= {f"{e.env_prefix}{suffix}" for e in config.settings.engines for suffix in ("_BASE_URL", "_API_KEY")}
    compose = set()
    for name in ("docker-compose.yml", "docker-compose.cpu.yml"):
        compose |= set(_ENV_IN_COMPOSE.findall((ROOT / name).read_text()))
    example = set(_ENV_IN_EXAMPLE.findall((ROOT / ".env.example").read_text()))
    return {"code": code, "compose": compose, "example": example, "fixed": _fixed()}


# compose's own tags (`!reset`, `!override`) read as the plain value they carry
class _ComposeLoader(yaml.SafeLoader):
    pass


_ComposeLoader.add_multi_constructor(
    "!",
    lambda loader, suffix, node: (
        loader.construct_mapping(node)
        if isinstance(node, yaml.MappingNode)
        else loader.construct_sequence(node)
        if isinstance(node, yaml.SequenceNode)
        else loader.construct_scalar(node)
    ),
)


# names an image or compose sets with a literal value: nobody sets them in .env
def _fixed() -> set[str]:
    names = set()
    for dockerfile in [ROOT / "Dockerfile", *sorted((ROOT / "converters").glob("*/Dockerfile"))]:
        # an ENV instruction runs on over backslashed lines and sets several names at once
        text = dockerfile.read_text().replace("\\\n", " ")
        for line in text.splitlines():
            if re.match(r"^\s*ENV\s", line):
                names |= set(re.findall(r"\b([A-Z][A-Z0-9_]+)=", line)) | set(
                    re.findall(r"^\s*ENV\s+([A-Z][A-Z0-9_]+)\s", line)
                )
    for name in ("docker-compose.yml", "docker-compose.cpu.yml"):
        doc = yaml.load((ROOT / name).read_text(), Loader=_ComposeLoader) or {}
        for service in (doc.get("services") or {}).values():
            env = (service or {}).get("environment") or {}
            items = env.items() if isinstance(env, dict) else [tuple((e.split("=", 1) + [None])[:2]) for e in env]
            # a bare name passes the host's value through; a literal is set here and nowhere else
            names |= {str(k) for k, v in items if v is not None and "${" not in str(v)}
    return names


def _literal(node) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        return True
    return isinstance(node, ast.BinOp) and _literal(node.left) and _literal(node.right)


def _numbers_in(node) -> int:
    return sum(1 for n in ast.walk(node) if _literal(n) and isinstance(n, ast.Constant))


def _regex(value) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "compile"
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "re"
    )


# module constants that are a number, a pattern or a container of numbers, and defaults that are a threshold or seed
def constants() -> list[str]:
    found = []
    for source in _walked_files():
        rel = source.relative_to(ROOT)
        tree = ast.parse(source.read_text())
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or not re.fullmatch(r"_?[A-Z][A-Z0-9_]*", target.id):
                continue
            value = node.value
            container = isinstance(value, (ast.Tuple, ast.List, ast.Dict, ast.Set)) and _numbers_in(value) > 0
            if _literal(value) or _regex(value) or container:
                found.append(f"{rel}:{target.id}")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args.args + node.args.kwonlyargs
            defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + node.args.defaults
            defaults += node.args.kw_defaults
            for arg, default in zip(args, defaults, strict=False):
                if default is None or not _literal(default) or not _DECIDING_PARAM.search(arg.arg):
                    continue
                if isinstance(default, ast.Constant) and default.value in (0, 1, -1):
                    continue
                found.append(f"{rel}:{node.name}({arg.arg})")
    return found


def tree() -> dict[str, list[str]]:
    env = env_sets()
    return {
        "config": list(config_keys()),
        "constant": constants(),
        "env": sorted(env["code"] | env["compose"] | env["example"]),
        "file": sorted(FILES),
    }


def _verdicts() -> dict:
    return yaml.safe_load(VERDICTS.read_text()) if VERDICTS.exists() else {}


# what is in the tree and not judged, and what is judged and no longer in the tree
def differences(found: dict, judged: dict) -> list[str]:
    out = []
    for kind, names in found.items():
        have = set((judged.get(kind) or {}).keys())
        out += [f"{kind} not judged: {n}" for n in names if n not in have]
        out += [f"{kind} judged but gone: {n}" for n in sorted(have - set(names))]
    families = judged.get("families") or {}
    for kind in found:
        for name, row in (judged.get(kind) or {}).items():
            if (row or {}).get("class") not in CLASSES:
                out.append(f"{kind} {name}: class must be one of {CLASSES}")
            if (row or {}).get("family") and row["family"] not in families:
                out.append(f"{kind} {name}: family {row['family']} is not declared")
    return out


def render(found: dict, judged: dict) -> str:
    annotations = config_annotations()
    env = env_sets()
    lines = [
        "# Config inventory",
        "",
        "Generated by `scripts/config_inventory.py` from the tree and `docs/config_inventory.yaml`; do not edit.",
        "Every rule or number that decides behaviour, where it lives, how it was chosen and whether it is a setting,",
        "code, a cap or a seed. A key the tree has and the verdict file does not fails `--check`.",
        "",
    ]
    where = config_keys()
    said = {k for k, r in (judged.get("config") or {}).items() if (r or {}).get("how")} | set(annotations)
    classes = {k: ((judged.get("config") or {}).get(k) or {}).get("class") for k in found["config"]}
    settings = [k for k in found["config"] if classes[k] == "setting"]
    unsaid = [k for k in settings if k not in said]
    seeds = [k for k in found["config"] if classes[k] == "seed"]
    totals = {k: len(v) for k, v in found.items()}
    lines += [
        f"Counted: {totals['config']} config keys, {totals['constant']} module constants, "
        f"{totals['env']} environment variables, {totals['file']} kinds of settings files.",
        f"The walk reads {', '.join(WALKED)}; a cloud engine's prefix lives in the database, not in the tree.",
        "",
        f"**Config settings with no word on how they were chosen: {len(unsaid)} of {len(settings)}.**",
        f"Seeds, where the database owns the value and the question does not apply: {len(seeds)}.",
        "",
    ]
    fixed = env["fixed"]
    missing = {
        "set by an image or by compose with a literal value": sorted(fixed),
        "read by code, not in .env.example": sorted(env["code"] - env["example"] - fixed),
        "in compose, not in .env.example": sorted(env["compose"] - env["example"] - fixed),
        "in .env.example, read by neither (a cloud prefix is read through its engine row)": sorted(
            env["example"] - env["code"] - env["compose"]
        ),
    }
    lines += ["## Environment sets", ""] + [f"- {k}: {', '.join(v) or 'none'}" for k, v in missing.items()] + [""]
    declared = judged.get("families") or {}
    members: dict[str, list[str]] = {fid: [] for fid in declared}
    for kind in found:
        for n, r in (judged.get(kind) or {}).items():
            if (r or {}).get("family") in members:
                members[r["family"]].append(n)
    held = sum(len(v) for v in members.values())
    lines += ["## One value in more than one holder", "", f"{len(declared)} families over {held} rows.", ""]
    lines += [f"- **{fid}**: {declared[fid]} ({', '.join(f'`{n}`' for n in members[fid])})" for fid in declared] + [""]
    for kind in found:
        lines += [
            f"## {kind}",
            "",
            "| name | where | class | how chosen | should move | note |",
            "|---|---|---|---|---|---|",
        ]
        for name in found[kind]:
            row = (judged.get(kind) or {}).get(name) or {}
            how = row.get("how") or annotations.get(name, "")
            here = where.get(name, "") if kind == "config" else ""
            move = row.get("move", "")
            lines.append(f"| `{name}` | {here} | {row.get('class', '')} | {how} | {move} | {row.get('note', '')} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    found = tree()
    judged = _verdicts()
    if "--list" in sys.argv:
        for kind, names in found.items():
            for n in names:
                print(kind, n)
        raise SystemExit(0)
    gaps = differences(found, judged)
    if "--check" in sys.argv:
        drawn = render(found, judged)
        stale = not TABLE.exists() or TABLE.read_text() != drawn
        for g in gaps:
            print(g)
        if stale:
            print(f"{TABLE.relative_to(ROOT)} is out of date, run scripts/config_inventory.py")
        raise SystemExit(1 if gaps or stale else 0)
    TABLE.write_text(render(found, judged))
    print(f"wrote {TABLE.relative_to(ROOT)}; {len(gaps)} gaps")
