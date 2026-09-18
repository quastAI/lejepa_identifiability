"""Boots `SimulationApp` -- the only module permitted to (README §4.2, §4.4).

Every other module under `idtb.sim` assumes this has already run in the
current process before any of its methods are called. `SimulationApp` is
process-global and one-shot: never call :func:`launch` a second time.
"""

from __future__ import annotations

import sys
from typing import Any


def launch(args: Any) -> Any:
    """`AppLauncher(args).app`.

    `args` is whatever `argparse.Namespace` the caller built after calling
    `AppLauncher.add_app_launcher_args(parser)` on its own parser (README
    §4.4) -- this function does not construct the parser, so a caller's own
    CLI flags pass through untouched.

    Sensor rendering in a standalone script needs `enable_cameras=True`
    (README §4.4); set here via `setattr` if the caller's parser exposed the
    flag but left it at its default, so a caller does not have to remember
    this every time.

    **Confirmed on the pod:** `AppLauncher`/`SimulationApp` hands Kit's own
    native CLI parser `sys.argv` directly, independent of `args` -- every
    prior Isaac-facing script here ran as `isaaclab.sh -p script.py [flags
    Kit already understands]`, so this never surfaced before. Booting from
    inside a `pytest` process leaves `sys.argv` full of pytest's own flags
    (e.g. `-m isaac`), which Kit's parser cannot parse and which crashed the
    whole process with a segfault rather than a catchable exception. `argv`
    is scrubbed to just the program name for the duration of the boot call,
    since Kit never needs more than that once `args` has already been built.
    """
    from isaaclab.app import AppLauncher

    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True
    saved_argv = sys.argv
    sys.argv = saved_argv[:1]
    try:
        return AppLauncher(args).app
    finally:
        sys.argv = saved_argv
