"""Backend factories for the tier-1 contract suite (README §10.3).

Every test in this package is written as a function of ``backend`` so a
future Isaac-backed :class:`idtb.sim.backend.SceneBackend` costs one entry in
``_BACKEND_FACTORIES``, not a rewrite of the suite -- the whole point of the
seam (README §4.3). Only :class:`idtb.sim.mock.MockSceneBackend` exists today:
docs/PLAN.md Phase 4 scoped an Isaac-facing backend out, since it would have
to either fake write paths for `style` knobs no spike has verified
(`table.roughness`/`table.albedo`) or one that is confirmed blocked
(`light.azimuth`/`light.elevation`, README §7.5) -- exactly the "resolve,
don't guess" discipline the spikes were written under.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from idtb.sim import MockSceneBackend, SceneBackend

_BACKEND_FACTORIES: dict[str, Callable[[], SceneBackend]] = {
    "mock": MockSceneBackend,
}


@pytest.fixture(params=list(_BACKEND_FACTORIES))
def backend(request: pytest.FixtureRequest) -> Iterator[SceneBackend]:
    made = _BACKEND_FACTORIES[request.param]()
    yield made
    made.close()
