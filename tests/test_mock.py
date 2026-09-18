"""``MockSceneBackend`` (README §4.3): deterministic and injective by
construction, numpy only, no Isaac import."""

from __future__ import annotations

import pytest
import torch

from idtb.latents import Handle, LatentSpec
from idtb.sim import MockSceneBackend, SceneBackend, UnsupportedRoleError


def _mid_phi(spec: LatentSpec, batch: int = 1) -> torch.Tensor:
    """Every handle at its own centre -- squash(0)."""
    return spec.squash(torch.zeros(batch, spec.n))


def test_mock_satisfies_the_scene_backend_protocol():
    assert isinstance(MockSceneBackend(), SceneBackend)


def test_bind_rejects_a_role_it_cannot_write():
    spec = LatentSpec((Handle("no.such.role", 0.0, 1.0),))
    backend = MockSceneBackend()
    with pytest.raises(UnsupportedRoleError):
        backend.bind(spec)


def test_handles_reports_write_path_per_role(full_style_spec: LatentSpec):
    backend = MockSceneBackend()
    backend.bind(full_style_spec.subset("base"))
    info = backend.handles()
    assert info["arm.j0"].write_path == "joint"
    assert info["cube.x"].write_path == "root_state"


def test_write_state_rejects_the_wrong_width(full_style_spec: LatentSpec):
    backend = MockSceneBackend()
    backend.bind(full_style_spec)
    with pytest.raises(ValueError, match="shape"):
        backend.write_state(torch.zeros(1, full_style_spec.n - 1))


def test_read_state_is_exact_by_construction(full_style_spec: LatentSpec):
    """The mock has no solver tolerance to lose to -- read-back is bitwise."""
    backend = MockSceneBackend()
    backend.bind(full_style_spec)
    phi = _mid_phi(full_style_spec, batch=3)
    backend.write_state(phi)
    assert torch.equal(backend.read_state(), phi)


def test_render_before_write_raises(full_style_spec: LatentSpec):
    backend = MockSceneBackend()
    backend.bind(full_style_spec)
    with pytest.raises(RuntimeError):
        backend.render(0)


def test_render_shapes_and_dtypes(full_style_spec: LatentSpec):
    backend = MockSceneBackend(resolution=(32, 40))
    backend.bind(full_style_spec)
    backend.write_state(_mid_phi(full_style_spec, batch=2))
    frame = backend.render(0)
    rgb, seg = frame["cam0"]["rgb"], frame["cam0"]["seg"]
    assert rgb.shape == (2, 32, 40, 3)
    assert rgb.dtype == torch.uint8
    assert seg.shape == (2, 32, 40)
    assert seg.dtype == torch.int64


def test_render_never_aliases_across_calls(full_style_spec: LatentSpec):
    """The mock is the correctness gates' known-good reference (README §4.3) --
    it must not reproduce Isaac's own aliased-buffer finding (§7.2)."""
    backend = MockSceneBackend()
    backend.bind(full_style_spec)
    backend.write_state(_mid_phi(full_style_spec))
    first = backend.render(0)["cam0"]["rgb"].clone()
    backend.render(1)
    assert torch.equal(first, backend.render(0)["cam0"]["rgb"])


def test_same_state_renders_bitwise_identical(full_style_spec: LatentSpec):
    backend = MockSceneBackend()
    backend.bind(full_style_spec)
    phi = _mid_phi(full_style_spec)
    backend.write_state(phi)
    a = backend.render(0)["cam0"]["rgb"].clone()
    backend.write_state(phi)
    b = backend.render(1)["cam0"]["rgb"].clone()
    assert torch.equal(a, b)


def test_diagnostics_before_render_raises(full_style_spec: LatentSpec):
    backend = MockSceneBackend()
    backend.bind(full_style_spec)
    backend.write_state(_mid_phi(full_style_spec))
    with pytest.raises(RuntimeError):
        backend.diagnostics()


@pytest.mark.parametrize("role", ["cube.x", "cube.hue", "light.intensity", "arm.j0"])
def test_every_group_s_roles_move_pixels(role: str, full_style_spec: LatentSpec):
    """Sensitivity per README §10.1 -- one dimension away from centre must
    change the render, for a representative role from each group."""
    backend = MockSceneBackend()
    backend.bind(full_style_spec)
    handle = full_style_spec.handles[full_style_spec.index(role)]

    base_phi = _mid_phi(full_style_spec)
    backend.write_state(base_phi)
    base_frame = backend.render(0)["cam0"]["rgb"].clone().float()

    moved_phi = base_phi.clone()
    moved_phi[0, full_style_spec.index(role)] = handle.center + 0.8 * handle.radius
    backend.write_state(moved_phi)
    moved_frame = backend.render(1)["cam0"]["rgb"].clone().float()

    assert (base_frame - moved_frame).abs().mean() > 0.0


def test_inactive_handles_are_held_at_their_centre(full_style_spec: LatentSpec):
    """README §5.2: a narrower active spec renders the same as the full spec
    with every excluded dimension sitting at its own handle centre."""
    base_backend = MockSceneBackend()
    base_spec = full_style_spec.subset("base")
    base_backend.bind(base_spec)
    base_backend.write_state(_mid_phi(base_spec))
    base_frame = base_backend.render(0)["cam0"]["rgb"]

    full_backend = MockSceneBackend()
    full_backend.bind(full_style_spec)
    full_backend.write_state(_mid_phi(full_style_spec))
    full_frame = full_backend.render(0)["cam0"]["rgb"]

    assert torch.equal(base_frame, full_frame)


def test_visibility_drops_when_the_arm_marker_covers_the_cube(full_style_spec: LatentSpec):
    """README §5.3: occlusion is measurable, not assumed away."""
    backend = MockSceneBackend()
    backend.bind(full_style_spec)

    apart = _mid_phi(full_style_spec)
    apart[0, full_style_spec.index("cube.x")] = 0.3
    apart[0, full_style_spec.index("arm.j0")] = -1.5
    backend.write_state(apart)
    backend.render(0)
    visible = backend.diagnostics()["visibility"].item()

    overlapping = _mid_phi(full_style_spec)
    backend.write_state(overlapping)
    backend.render(1)
    concentric = backend.diagnostics()["visibility"].item()

    assert visible > concentric
