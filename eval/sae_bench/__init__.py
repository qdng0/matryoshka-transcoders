"""
Local SAE Bench-compatible package for Matryoshka evals.

Extends the namespace to include the upstream SAEBench copy so both
sae_bench.absorption (local overrides) and sae_bench.evals.* (upstream)
can be imported together.
"""

from pathlib import Path
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]

_root = Path(__file__).resolve().parents[2]
_saebench_pkg = _root / "SAEBench" / "sae_bench"
if _saebench_pkg.exists():
    __path__.append(str(_saebench_pkg))


