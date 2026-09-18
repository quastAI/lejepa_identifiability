"""Tier-1 contract suite (README §10.1, §10.3).

One set of tests, run as a function of ``backend`` (``tests/contract/conftest.py``)
and ``group_spec`` (``tests/conftest.py`` -- ``base``, ``base+style``, ``full``,
``full+style``). A gate only ever observed against one implementation is weak
evidence (README §4.3); this is what makes a future pass against a real Isaac
backend meaningful rather than merely reassuring.
"""

from __future__ import annotations

import pytest
import torch

from idtb.gates import determinism_gate, read_back_gate, style_sensitivity_gate
from idtb.latents import LatentSpec, sample_ou_pairs
from idtb.sim.backend import WRITE_PATHS, SceneBackend


def _mid_phi(spec: LatentSpec, batch: int = 1) -> torch.Tensor:
    return spec.squash(torch.zeros(batch, spec.n))


def _capture(backend: SceneBackend):
    def capture(phi: torch.Tensor) -> torch.Tensor:
        backend.write_state(phi)
        return backend.render(0)["cam0"]["rgb"]

    return capture


def test_bind_declares_every_role_with_a_known_write_path(
    backend: SceneBackend, group_spec: LatentSpec
):
    backend.bind(group_spec)
    info = backend.handles()
    assert set(info) == set(group_spec.roles)
    assert all(h.write_path in WRITE_PATHS for h in info.values())


def test_write_then_read_back_matches_to_tolerance(backend: SceneBackend, group_spec: LatentSpec):
    """README §6.3's hard gate -- non-negotiable, run here against every
    group configuration, not only ``base``."""
    backend.bind(group_spec)
    z, _ = sample_ou_pairs(
        group_spec.n, batch=2, rho=0.9, generator=torch.Generator().manual_seed(0)
    )
    phi = group_spec.squash(z)
    backend.write_state(phi)
    read_back_gate(phi, backend.read_state(), atol=1e-4)


def test_render_output_shape_and_dtype(backend: SceneBackend, group_spec: LatentSpec):
    backend.bind(group_spec)
    backend.write_state(_mid_phi(group_spec, batch=2))
    frame = backend.render(0)
    assert frame, "render() must return at least one camera"
    for cam in frame.values():
        assert cam["rgb"].dtype == torch.uint8
        assert cam["rgb"].shape[0] == 2
        assert cam["rgb"].shape[-1] == 3
        assert cam["seg"].dtype == torch.int64
        assert cam["seg"].shape[0] == 2


def test_diagnostics_present_and_in_range(backend: SceneBackend, group_spec: LatentSpec):
    backend.bind(group_spec)
    backend.write_state(_mid_phi(group_spec, batch=3))
    backend.render(0)
    diagnostics = backend.diagnostics()
    assert "visibility" in diagnostics
    assert "collision" in diagnostics
    assert diagnostics["visibility"].shape[0] == 3
    assert bool(((diagnostics["visibility"] >= 0) & (diagnostics["visibility"] <= 1)).all())


def test_render_is_deterministic(backend: SceneBackend, group_spec: LatentSpec):
    """README §7.1's acceptance test -- bitwise, run against every group
    configuration this backend is bound to."""
    backend.bind(group_spec)
    capture = _capture(backend)
    base = _mid_phi(group_spec)
    other = base.clone()
    other[0, 0] = group_spec.handles[0].center + 0.5 * group_spec.handles[0].radius

    determinism_gate(capture, base, other)


def test_distinct_states_do_not_render_identically(backend: SceneBackend, group_spec: LatentSpec):
    """Injectivity proxy (README §10.1): two well-separated latents must not
    collapse to the same pixels."""
    backend.bind(group_spec)
    phi_a = group_spec.squash(torch.zeros(1, group_spec.n))
    phi_b = group_spec.squash(torch.ones(1, group_spec.n))

    backend.write_state(phi_a)
    frame_a = backend.render(0)["cam0"]["rgb"].clone()
    backend.write_state(phi_b)
    frame_b = backend.render(1)["cam0"]["rgb"].clone()

    assert not torch.equal(frame_a, frame_b)


def test_style_knobs_are_all_sensitive(backend: SceneBackend, group_spec: LatentSpec):
    """README §7.5/§10.1: a disconnected `style` write must not pass unnoticed
    in *any* group configuration that activates `style`."""
    style_dims = group_spec.dims("style")
    if not style_dims:
        pytest.skip("no style dims active in this group configuration")

    backend.bind(group_spec)
    capture = _capture(backend)
    base = _mid_phi(group_spec)
    perturbations = {}
    for dim in style_dims:
        handle = group_spec.handles[dim]
        phi = base.clone()
        phi[0, dim] = handle.center + 0.8 * handle.radius
        perturbations[handle.role] = phi

    style_sensitivity_gate(capture, base, perturbations, noise_floor=0.0)
