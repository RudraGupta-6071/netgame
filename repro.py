"""
repro.py -- reproducibility metadata for experiment outputs.

Every experiment run writes `out/RUN_METADATA.json` recording what produced the
numbers: interpreter and package versions, the repository commit, the platform
and CPU (runtime figures are hardware dependent and are NOT reproducible across
machines), the random seeds used, the runtime methodology, and the full
tolerance configuration from `tolerances.py`.

This matters most for the randomised validation (`tests.py`), the large
path-generation experiments (E7) and any numerical certificate claim: without
the tolerance set and the commit, a reported gap cannot be checked.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

import tolerances

HERE = os.path.dirname(os.path.abspath(__file__))

__all__ = ["package_versions", "git_commit", "run_metadata", "write_metadata",
           "SEEDS", "RUNTIME_METHODOLOGY"]

#: Every random seed used anywhere in the study, by consumer.  Seeding is
#: explicit throughout, so the numerical results regenerate exactly; measured
#: wall-clock times do not.
SEEDS = {
    "graphs.random_dag (CATALOG random-12)": 7,
    "experiments.E5_baselines (20 random DAGs)": list(range(20)),
    "experiments.E6_convergence (random-12)": 7,
    "experiments.E7_scaling (random DAG sizes, 3 repeats)": [100, 101, 102],
    "tests.test_inner_vs_random_sampling": 0,
    "tests.test_inner_kkt_and_budget": 1,
    "tests.test_dual_bound": 2,
    "tests.test_danskin": 4,
    "tests.test_convexity": 5,
    "tests.test_oracle_equivalence": list(range(10)),
    "tests.test_heuristics_never_better": list(range(6)),
    "tests.test_general_m": [8, 9],
    "tests.test_contested_endpoints": [12, 3, list(range(6))],
    "tests.test_uniqueness": 11,
    "tests.test_m1_closed_form": 21,
    "tests.test_zero_defence_boundary": 22,
    "tests.test_lambda_search_adaptive": 23,
}

RUNTIME_METHODOLOGY = (
    "Wall-clock seconds measured with time.time() around a single solve, on a "
    "warm interpreter, with no repetition and no averaging unless a column is "
    "explicitly named mean_sec (mean of the stated number of seeds). Timings "
    "include path-oracle construction unless an oracle is passed in. They are "
    "machine dependent and are reported as observed performance on the stated "
    "hardware, not as a complexity result."
)


def package_versions():
    """Versions of the packages whose numerics affect the results."""
    out = {"python": sys.version.split()[0],
           "python_full": sys.version.replace("\n", " ")}
    for mod in ("numpy", "scipy", "networkx", "matplotlib"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as exc:                        # pragma: no cover
            out[mod] = f"unavailable ({type(exc).__name__})"
    return out


def git_commit():
    """Repository commit that generated the results (or a clear marker)."""
    info = {"commit": "unknown", "dirty": None, "branch": None}
    try:
        info["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
            text=True, timeout=10).stdout.strip() or "unknown"
        info["branch"] = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=HERE,
            capture_output=True, text=True, timeout=10).stdout.strip() or None
        status = subprocess.run(["git", "status", "--porcelain"], cwd=HERE,
                                capture_output=True, text=True, timeout=10)
        info["dirty"] = bool(status.stdout.strip())
    except Exception as exc:                            # pragma: no cover
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def hardware():
    """Platform / CPU description, recorded because runtimes are reported."""
    return {"platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count()}


def run_metadata(extra=None):
    """The full metadata record written alongside experiment outputs."""
    meta = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository": git_commit(),
        "versions": package_versions(),
        "hardware": hardware(),
        "runtime_methodology": RUNTIME_METHODOLOGY,
        "seeds": SEEDS,
        "tolerances": tolerances.as_dict(),
    }
    if extra:
        meta.update(extra)
    return meta


def write_metadata(out_dir, name="RUN_METADATA.json", extra=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as fh:
        json.dump(run_metadata(extra), fh, indent=2, default=str)
    return path


if __name__ == "__main__":
    print(json.dumps(run_metadata(), indent=2, default=str))
