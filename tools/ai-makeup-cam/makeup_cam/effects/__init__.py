"""Makeup effects.

Effect classes register themselves when their module is imported. The
pipeline calls ``load_builtin_effects()`` to import every built-in module;
importing ``makeup_cam.effects.base`` alone stays lightweight.
"""

from __future__ import annotations

import importlib

from .base import Effect, FrameContext, register_effect, registered_effects

# Built-in effect modules (execution order is set by each Effect.order).
BUILTIN_EFFECT_MODULES = ("reshape", "hairshape", "hair", "skin", "contour", "brows", "eyes", "blush", "lips", "teeth")


def load_builtin_effects() -> list[type[Effect]]:
    """Import all built-in effect modules and return the registered classes."""
    for module in BUILTIN_EFFECT_MODULES:
        importlib.import_module(f"{__name__}.{module}")
    return registered_effects()


__all__ = [
    "BUILTIN_EFFECT_MODULES",
    "Effect",
    "FrameContext",
    "load_builtin_effects",
    "register_effect",
    "registered_effects",
]
