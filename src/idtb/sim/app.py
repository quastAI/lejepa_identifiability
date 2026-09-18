"""Boots `SimulationApp` -- the only module permitted to (README §4.2, §4.4).

Every other module under `idtb.sim` assumes this has already run in the
current process before any of its methods are called. `SimulationApp` is
process-global and one-shot: never call :func:`launch` a second time.
"""

from __future__ import annotations

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
    """
    from isaaclab.app import AppLauncher

    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True
    return AppLauncher(args).app
