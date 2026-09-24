"""Reproducible, conservative migration inventory. Development tool, never a runtime.

The scanner records evidence, including unresolved dispatch, rather than asserting
that static references prove a live service call. Every production function stays
in the denominator, even if no numerical syntax is detected.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

SCHEMA = "platform-compute-inventory-1"
EXCLUDED = {"tests", "scripts", "__pycache__"}
COMPILERS = {"njit", "jit", "vectorize", "guvectorize", "cfunc"}
NUMERICAL_METHODS = {
    "sum", "mean", "std", "var", "min", "max", "median", "quantile", "percentile",
    "cumsum", "cumprod", "diff", "pct_change", "rolling", "groupby", "dot", "matmul",
    "solve", "eigh", "eig", "svd", "cholesky", "norm", "exp", "log", "sqrt",
    "rank", "corr", "cov", "clip", "linspace", "arange", "interp", "polyfit",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def symbol(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = symbol(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return symbol(node.func)
    return ""


def imported_name(text: str, aliases: dict[str, str]) -> str:
    first, _, rest = text.partition(".")
    return aliases.get(first, first) + (f".{rest}" if rest else "")


def local_nodes(node: ast.AST):
    """Do not credit a wrapper with computations belonging to nested functions."""
    yield node
    children = node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else ast.iter_child_nodes(node)
    for child in children:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield from local_nodes(child)


def import_aliases(module: str, path: str, nodes) -> dict[str, str]:
    aliases = {}
    for item in nodes:
        if isinstance(item, ast.Import):
            for alias in item.names:
                aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(item, ast.ImportFrom):
            base = item.module or ""
            if item.level:
                parts = module.split(".")
                if Path(path).name != "__init__.py":
                    parts.pop()
                base = ".".join(parts[:len(parts) - item.level + 1] + ([base] if base else []))
            for alias in item.names:
                aliases[alias.asname or alias.name] = f"{base}.{alias.name}"
    return aliases


def definitions(tree: ast.AST, prefix: str = ""):
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = f"{prefix}.{node.name}" if prefix else node.name
            if not isinstance(node, ast.ClassDef):
                yield name, node
            yield from definitions(node, name)
        else:
            yield from definitions(node, prefix)


def phase(path: str) -> str:
    if any(p in path for p in ("qp_numba", "solver")):
        return "M3"
    if any(p in path for p in ("optimizer", "frontier", "strategy.py", "fit_numba")):
        return "M4"
    if any(p in path for p in ("auto_class", "historical_regimes")):
        return "M5"
    if any(p in path for p in ("scenario_stress", "product_analysis_numba", "synthetic")):
        return "M6"
    if any(p in path for p in ("strategic_allocation", "pre_investment")):
        return "M7"
    if any(p in path for p in ("cal_indicators", "numeric", "numba_warmup")):
        return "M2"
    return "M8"


def scan(root: Path) -> dict:
    root = root.resolve()
    commit = git(root, "rev-parse", "HEAD")
    paths = sorted(p for p in git(root, "ls-files", "-z", "*.py").split("\0") if p)
    production = [p for p in paths if not EXCLUDED.intersection(Path(p).parts)]
    trees, files, functions, all_names = {}, [], {}, defaultdict(list)
    for path in production:
        raw = (root / path).read_bytes()
        tree = ast.parse(raw, filename=path)
        trees[path] = tree
        module = path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        files.append({"path": path, "sha256": digest(raw), "module": module})
        for name, node in definitions(tree):
            key = f"{path}::{name}"
            # Python permits redefining a function. Keep both instead of silently
            # overwriting one possible numerical implementation.
            if key in functions:
                previous = functions.pop(key)
                old_key = f"{key}@{previous[1].lineno}"
                functions[old_key] = previous
                for names in all_names.values():
                    names[:] = [old_key if x == key else x for x in names]
                key = f"{key}@{node.lineno}"
            elif all_names[f"{module}.{name}"]:
                key = f"{key}@{node.lineno}"
            functions[key] = (path, node, name)
            all_names[f"{module}.{name}"].append(key)
            if module.startswith("backend."):
                all_names[f"{module[8:]}.{name}"].append(key)
    rows, dynamic_sites = [], []
    for key, (path, node, name) in sorted(functions.items()):
        module = path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        tree = trees[path]
        # Module imports plus function-local imports; imports inside unrelated
        # functions must not shadow this function's binding.
        imports = list(local_nodes(tree)) + list(local_nodes(node))
        aliases = import_aliases(module, path, imports)

        def resolve(text):
            first, _, rest = text.partition(".")
            targets = []
            if first in aliases:
                targets += all_names.get(aliases[first] + (f".{rest}" if rest else ""), [])
            scope = name.split(".")[:-1]
            for depth in range(len(scope), -1, -1):
                scoped = ".".join([module, *scope[:depth], text])
                targets += all_names.get(scoped, [])
            if first in {"self", "cls"} and scope:
                targets += all_names.get(".".join([module, *scope, rest]), [])
            return sorted(set(targets))

        nodes = list(local_nodes(node))
        calls = []
        references = set()
        numerical = set()
        for item in nodes:
            if isinstance(item, (ast.Name, ast.Attribute)) and isinstance(item.ctx, ast.Load):
                references.update(resolve(symbol(item)))
            if isinstance(item, (ast.BinOp, ast.AugAssign)):
                numerical.add("arithmetic_syntax")
            if isinstance(item, ast.Call):
                target = symbol(item.func)
                resolved = resolve(target)
                calls.append({"line": item.lineno, "target": target or "<dynamic>", "resolved": resolved})
                leaf = imported_name(target, aliases).rsplit(".", 1)[-1]
                if leaf in COMPILERS or leaf in {"exec", "eval", "compile"}:
                    dynamic_sites.append({"source_id": key, "line": item.lineno, "call": ast.unparse(item)})
                if leaf in NUMERICAL_METHODS or target.startswith(("np.linalg.", "np.random.", "math.")):
                    numerical.add(f"numerical_call:{leaf}")
        decorators = [ast.unparse(d) for d in node.decorator_list]
        decorated = any(imported_name(symbol(d), aliases).rsplit(".", 1)[-1] in COMPILERS
                        for d in node.decorator_list)
        if decorated:
            numerical.add("compiled_decorator")
        defaults = [None] * (len(node.args.posonlyargs) + len(node.args.args) - len(node.args.defaults))
        defaults += [ast.unparse(v) for v in node.args.defaults]
        arguments = [
            {"name": arg.arg, "annotation": ast.unparse(arg.annotation) if arg.annotation else None, "default": default}
            for arg, default in zip([*node.args.posonlyargs, *node.args.args], defaults, strict=True)
        ]
        arguments += [
            {"name": arg.arg, "annotation": ast.unparse(arg.annotation) if arg.annotation else None,
             "default": ast.unparse(default) if default is not None else None, "keyword_only": True}
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
        ]
        rows.append({
            "source_id": key, "source_path": path, "symbol": name,
            "line": node.lineno, "end_line": node.end_lineno,
            "ast_sha256": digest(ast.dump(node, include_attributes=False).encode()),
            "decorators": decorators, "compiled_decorator": decorated,
            "arguments": arguments,
            "variadic": {"args": node.args.vararg.arg if node.args.vararg else None,
                         "kwargs": node.args.kwarg.arg if node.args.kwarg else None},
            "return_annotation": ast.unparse(node.returns) if node.returns else None,
            "contract_source": f"{path}:{node.lineno}",
            "numeric_evidence": sorted(numerical),
            "classification": "compiled_numerical" if decorated else "numerical_candidate" if numerical else "orchestration_candidate",
            "calls": calls, "references": sorted(references),
            "proposed_phase": phase(path),
            "engine_equivalent": "not_verified", "platform_integrated": "not_verified",
            "service_verified": "not_verified", "speed_gate": "not_measured", "memory_gate": "not_measured",
        })
    # Capture module-level registrations/compilation independently of functions.
    module_references = []
    for path, tree in trees.items():
        module = path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        aliases = import_aliases(module, path, local_nodes(tree))
        for item in local_nodes(tree):
            if isinstance(item, ast.Call) and imported_name(symbol(item.func), aliases).rsplit(".", 1)[-1] in COMPILERS | {"exec", "eval", "compile"}:
                dynamic_sites.append({"source_id": f"{path}::<module>", "line": item.lineno, "call": ast.unparse(item)})
            if isinstance(item, (ast.Name, ast.Attribute)) and isinstance(item.ctx, ast.Load):
                module_references.append({"path": path, "line": item.lineno, "reference": symbol(item)})
    incoming = defaultdict(set)
    for row in rows:
        for target in row["references"]:
            incoming[target].add(row["source_id"])
    for row in rows:
        row["referenced_by"] = sorted(incoming[row["source_id"]])
        row["reachability"] = "static_reference" if row["referenced_by"] else "entry_or_dynamic_or_unreferenced"
    return {
        "schema": SCHEMA, "source_commit": commit, "source_tree": git(root, "rev-parse", "HEAD^{tree}"),
        "modified_tracked_python": sorted(set(git(root, "diff", "--name-only", "HEAD").splitlines()) & set(production)),
        "untracked_python": sorted(p for p in git(root, "ls-files", "--others", "--exclude-standard", "*.py").splitlines()
                                   if not EXCLUDED.intersection(Path(p).parts)),
        "scope": {"tracked": "*.py", "excluded_path_parts": sorted(EXCLUDED),
                  "evidence": "static_only", "unresolved_dispatch": "retained_not_excluded"},
        "files": files, "functions": rows,
        "dynamic_sites": sorted(dynamic_sites, key=lambda x: (x["source_id"], x["line"])),
        "module_references": sorted(module_references, key=lambda x: (x["path"], x["line"], x["reference"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path, help="Fail on any source/inventory drift, not just HEAD changes")
    args = parser.parse_args()
    result = scan(args.source)
    if args.check:
        old = json.loads(args.check.read_text())
        if old != result:
            raise SystemExit("Platform coverage inventory drift; regenerate and review the changed entries")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # One record per line keeps large inventories reviewable without 10 MB
        # of indentation. The artifact remains a single ordinary JSON object.
        sections = []
        for key, value in sorted(result.items()):
            rendered = ("[\n" + ",\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in value) + "\n]") if isinstance(value, list) else json.dumps(value, ensure_ascii=False, sort_keys=True)
            sections.append(json.dumps(key) + ": " + rendered)
        args.output.write_text("{\n" + ",\n".join(sections) + "\n}\n")
    print(json.dumps({"files": len(result["files"]), "functions": len(result["functions"]),
                      "decorated": sum(r["compiled_decorator"] for r in result["functions"]),
                      "dynamic_sites": len(result["dynamic_sites"]), "source_commit": result["source_commit"]}))


if __name__ == "__main__":
    main()
