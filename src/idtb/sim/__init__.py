"""The backend seam (README §4.3) and its implementations.

``backend.py`` is the pure protocol. Isaac-facing implementations import
``isaaclab``/``omni``/``carb``/``pxr`` lazily, inside functions, never at
module level (README §4.2) -- enforced by ``tests/test_import_guard.py`` and
ruff's TID253.
"""

from idtb.sim.backend import HandleInfo, SceneBackend, UnsupportedRoleError, WritePath
from idtb.sim.mock import MockSceneBackend

__all__ = [
    "HandleInfo",
    "MockSceneBackend",
    "SceneBackend",
    "UnsupportedRoleError",
    "WritePath",
]
