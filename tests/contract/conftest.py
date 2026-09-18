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
def _simulation_app() -> Iterator[None]:
    """Boots `SimulationApp` once for the whole tier-2/tier-1-Isaac session
    (README §4.2, §10.3) -- a second `AppLauncher` in this process is not
    supported. Only ever requested from the ``isaac`` branch of ``backend``,
    which is itself deselected by default, so a local run never reaches this.
    """
    import argparse

    from isaaclab.app import AppLauncher

    from idtb.sim.app import launch

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless"])
    app = launch(args)
    yield app
    app.close()


@pytest.fixture(
    params=[
        "mock",
        pytest.param("isaac", marks=pytest.mark.isaac),
    ]
)
def backend(request: pytest.FixtureRequest) -> Iterator[SceneBackend]:
    if request.param == "mock":
        made: SceneBackend = MockSceneBackend()
    else:
        # Lazily pulls in `_simulation_app` only for this branch, so a local
        # `pytest` run (which deselects `isaac`) never has to boot Isaac.
        request.getfixturevalue("_simulation_app")
        from idtb.sim import IsaacSceneBackend

        made = IsaacSceneBackend()
    yield made
    made.close()
