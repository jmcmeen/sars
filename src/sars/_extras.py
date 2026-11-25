"""Optional dependency groups ("extras") and the error raised when one is missing."""

from __future__ import annotations

import importlib.util

# Top-level modules each extra provides (import names, not distribution names).
EXTRAS: dict[str, tuple[str, ...]] = {
    "geo": ("geopandas", "shapely"),
    "server": ("fastapi", "uvicorn"),
    "bayes": ("pymc", "arviz"),
}


class MissingExtraError(ImportError):
    """An optional feature needs an extra that isn't installed."""


def install_hint(extra: str) -> str:
    """The command that installs ``extra``."""
    return f"pip install 'sars[{extra}]'"


def require_extra(extra: str, feature: str) -> None:
    """Raise `MissingExtraError` unless every module ``extra`` provides can be
    found (none are imported)."""
    missing = [m for m in EXTRAS[extra] if importlib.util.find_spec(m) is None]
    if missing:
        raise MissingExtraError(
            f"{feature} needs the [{extra}] extra (missing: {', '.join(missing)}). "
            f"Install it with: {install_hint(extra)}"
        )
