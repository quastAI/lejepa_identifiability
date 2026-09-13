"""Every module must import with the Isaac roots unavailable.

`SimulationApp` has to be constructed before `isaaclab`/`omni`/`carb`/`pxr` are
imported, so sim-touching modules import them inside functions (README §4.2). A
module-level import would otherwise surface as a pytest collection error on the
pod -- one container restart per violation -- and cannot fail locally at all,
because Isaac is not installable on macOS. Ruff's TID253 catches the literal
syntax; this catches everything else that pulls Isaac in at import time.
"""

import subprocess
import sys

SCRIPT = """
import importlib, pkgutil, sys

BLOCKED = {"isaacsim", "isaaclab", "omni", "carb", "pxr"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in BLOCKED:
            raise ImportError(f"module-level Isaac import of {name!r}")
        return None


sys.meta_path.insert(0, Blocker())

try:  # negative control: a gate nobody has seen fire is not a gate
    import omni  # noqa: F401
except ImportError:
    pass
else:
    raise SystemExit("blocker inactive")

import idtb

for module in pkgutil.walk_packages(idtb.__path__, "idtb."):
    importlib.import_module(module.name)
"""


def test_no_module_level_isaac_imports():
    result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
