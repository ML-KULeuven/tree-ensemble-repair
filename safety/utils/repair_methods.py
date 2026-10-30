"""Before submission milp was called fairness-gurobi and milp-l1 was called gurobi. This is there to preserve backwards compatibility for parsing results."""

from pathlib import Path
import re


def canonical_method_names(text):
    """Normalize method IDs in historical names, manifests, and model paths."""
    return re.sub(
        r"(?<![a-zA-Z0-9-])(fairness-gurobi|gurobi)(?=_|/|$)",
        lambda match: {"fairness-gurobi": "milp", "gurobi": "milp-l1"}[match[0]],
        text,
    )


def resolve_result_path(path):
    """Prefer the requested file; fall back to its historical method name.
    Use only when reading: new experiments must use new terminology.
    """
    path = Path(path)
    if path.exists():
        return path
    legacy_name = re.sub(
        r"(?<![a-zA-Z0-9-])(milp-l1|milp)(?=_|$)",
        lambda match: {"milp": "fairness-gurobi", "milp-l1": "gurobi"}[match[0]],
        path.name,
    )
    legacy_path = path.with_name(legacy_name)
    return legacy_path if legacy_path.exists() else path
