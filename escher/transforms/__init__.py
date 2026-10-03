"""escher.transforms -- the five paper transform families and their registry.

``FAMILIES`` maps each family name to a :class:`TransformFamily` instance
exposing a matched forward ``T`` and generalized inverse ``Tp``; ``get_family``
looks a name up with a clear error on typos. Family-specific parameters are
passed straight through to ``T``/``Tp`` as keyword arguments.
"""

from __future__ import annotations

from typing import Dict

from .base import TransformFamily, WARMUP_CONF_ZOOM
from .conformal import ConformalFamily
from .mobius import MobiusFamily
from .poles import PolesFamily
from .rimrings import RimringsFamily
from .square import SquareFamily

FAMILIES: Dict[str, TransformFamily] = {
    "conformal": ConformalFamily(),
    "poles": PolesFamily(),
    "mobius": MobiusFamily(),
    "square": SquareFamily(),
    "rimrings": RimringsFamily(),
}


def get_family(name: str) -> TransformFamily:
    try:
        return FAMILIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(FAMILIES))
        raise ValueError(f"unknown transform family {name!r}; known families: {known}") from exc


__all__ = ["FAMILIES", "get_family", "TransformFamily", "WARMUP_CONF_ZOOM"]
