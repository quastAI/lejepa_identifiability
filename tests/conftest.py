"""Shared fixtures for the pure/mock test tiers.

``FULL_STYLE_SPEC`` is the canonical spec :class:`idtb.sim.mock.MockSceneBackend`
renders end to end -- every role README §5.2 names, across all three groups.
Tests that need a narrower view use ``LatentSpec.subset`` on it directly, which
is exactly what a real run does (§6.4): one spec, one sampler, one writer.
"""

from __future__ import annotations

import pytest

from idtb.latents import Handle, LatentSpec

FULL_STYLE_SPEC = LatentSpec(
    (
        Handle("arm.j0", 0.0, 1.5, group="base"),
        Handle("arm.j1", 0.0, 1.0, group="base"),
        Handle("arm.j2", -1.5, 1.5, group="base"),
        Handle("arm.j3", 1.8, 1.8, group="base"),
        Handle("gripper.aperture", 0.02, 0.02, group="base"),
        Handle("cube.x", 0.0, 0.3, group="base"),
        Handle("cube.y", 0.0, 0.3, group="base"),
        Handle("cube.size", 0.05, 0.02, group="full"),
        Handle("cube.hue", 0.5, 0.45, group="full"),
        Handle("light.intensity", 0.0, 1.0, group="style"),
        Handle("light.warmth", 0.0, 1.0, group="style"),
        Handle("light.azimuth", 0.0, 1.0, group="style"),
        Handle("light.elevation", 0.0, 1.0, group="style"),
        Handle("cam.jitter.x", 0.0, 0.5, group="style"),
        Handle("cam.jitter.y", 0.0, 0.5, group="style"),
        Handle("table.roughness", 0.0, 1.0, group="style"),
        Handle("table.albedo", 0.0, 1.0, group="style"),
        Handle("exposure", 0.0, 1.0, group="style"),
    )
)


#: The four run configurations docs/PLAN.md Phase 4 asks the contract suite to
#: cover. ``full`` always implies ``base`` (§5.2's groups are cumulative);
#: ``style`` is disjoint and optional on top of either task group.
_GROUP_CONFIGS: dict[str, frozenset[str]] = {
    "base": frozenset({"base"}),
    "base+style": frozenset({"base", "style"}),
    "full": frozenset({"base", "full"}),
    "full+style": frozenset({"base", "full", "style"}),
}


@pytest.fixture
def full_style_spec() -> LatentSpec:
    return FULL_STYLE_SPEC


@pytest.fixture(params=list(_GROUP_CONFIGS))
def group_spec(request: pytest.FixtureRequest) -> LatentSpec:
    """One of the four group configurations, filtered from ``FULL_STYLE_SPEC``.

    Filtering (rather than ``LatentSpec.subset``, which is cumulative only up
    to a single named group) keeps the ``base``/``full``/``style`` ordering
    contract intact: dropping ``full`` while keeping ``style`` still leaves
    ranks non-decreasing.
    """
    wanted = _GROUP_CONFIGS[request.param]
    return LatentSpec(tuple(h for h in FULL_STYLE_SPEC.handles if h.group in wanted))
