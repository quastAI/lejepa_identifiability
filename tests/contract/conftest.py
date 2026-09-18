"""Backend factories for the tier-1 contract suite (README §10.3).

Every test in this package is written as a function of ``backend`` so a
second :class:`idtb.sim.backend.SceneBackend` costs one entry here, not a
rewrite of the suite -- the whole point of the seam (README §4.3).

``IsaacSceneBackend`` is registered behind ``@pytest.mark.isaac`` (deselected
by default, per ``pyproject.toml``'s ``-m "not isaac"``): it needs a real
``SimulationApp`` booted first (README §4.2), which only happens under
Isaac's own interpreter on the pod, and it is **written but not yet run**
(docs/PLAN.md Phase 4) -- combining a Franka arm with every verified
attribute write in one scene has never been exercised end to end before.
`light.azimuth`/`light.elevation` and `table.roughness` stay unregistered
anywhere: they are confirmed blocked (README §7.5, §11), not merely unbuilt,
so ``group_spec``'s ``base+style``/``full+style`` configurations only ever
exercise roles this backend actually declares support for.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from idtb.sim import MockSceneBackend, SceneBackend

_BACKEND_FACTORIES: dict[str, Callable[[], SceneBackend]] = {
    "mock": MockSceneBackend,
}


@pytest.fixture(scope="session")
def _simulation_app() -> None:
    """Boots `SimulationApp` once for the whole tier-2/tier-1-Isaac session
    (README §4.2, §10.3) -- a second `AppLauncher` in this process is not
    supported. Only ever requested from the ``isaac`` branch of ``backend``,
    which is itself deselected by default, so a local run never reaches this.

    **Deliberately no teardown -- never calls `app.close()`.** Confirmed on
    the pod twice: `SimulationApp.close()` can hard-terminate the process
    without returning control to Python (a standalone boot diagnostic script
    never printed its post-close line either). Doing that from a
    session-fixture teardown means the process dies before pytest's own
    terminal reporter gets to print anything -- every test result was
    already recorded (a run of `E`s on screen), but the summary and failure
    details never appeared because nothing survived long enough to print
    them. The process exits right after this session ends regardless, so
    there is nothing here that actually needs a clean close.
    """
    import argparse

    from isaaclab.app import AppLauncher

    from idtb.sim.app import launch

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless"])
    return launch(args)


@pytest.fixture(scope="session")
def _isaac_backend(_simulation_app: None) -> SceneBackend:
    """Built once per session and shared across every `isaac`-marked test.

    **Confirmed on the pod:** a function-scoped `IsaacSceneBackend()` --
    building a fresh `InteractiveScene` per test -- either collides with the
    prims the first one already spawned on the same live stage, or violates
    `SimulationContext` being a one-per-process object the same way
    `SimulationApp` is (README §4.2). The process died with no clean
    traceback after the first test's scene build succeeded. `bind()` on the
    shared instance is cheap and side-effect-free, so re-binding a different
    `group_spec` per test on the *same* scene is the correct level of reuse.
    """
    from idtb.sim import IsaacSceneBackend

    return IsaacSceneBackend()


@pytest.fixture(
    params=[
        "mock",
        pytest.param("isaac", marks=pytest.mark.isaac),
    ]
)
def backend(request: pytest.FixtureRequest) -> Iterator[SceneBackend]:
    if request.param == "mock":
        made: SceneBackend = MockSceneBackend()
        yield made
        made.close()
    else:
        # Lazily pulls in `_isaac_backend` only for this branch, so a local
        # `pytest` run (which deselects `isaac`) never has to boot Isaac.
        # Shared, session-scoped -- never closed per test (see
        # `_isaac_backend`'s docstring for why).
        yield request.getfixturevalue("_isaac_backend")
