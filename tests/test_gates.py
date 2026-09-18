"""Pipeline-correctness gates (README §10.1), exercised against
``MockSceneBackend`` -- library code, not test-only (docs/PLAN.md Phase 4).

Most of this file is **negative controls**. A gate that has only ever been
seen to pass is not a gate -- it is a detector nobody has watched detect
anything. Each fake below is a specific way the real pipeline could be
broken; the test asserts the matching gate *fires*, with a clear
:class:`idtb.gates.GateFailed` naming which check failed.
"""

from __future__ import annotations

import pytest
import torch

from idtb.gates import GateFailed, determinism_gate, read_back_gate, style_sensitivity_gate
from idtb.latents import LatentSpec
from idtb.sim import MockSceneBackend


def _mid_phi(spec: LatentSpec, batch: int = 1) -> torch.Tensor:
    return spec.squash(torch.zeros(batch, spec.n))


def _capture_factory(backend: MockSceneBackend):
    def capture(phi: torch.Tensor) -> torch.Tensor:
        backend.write_state(phi)
        return backend.render(0)["cam0"]["rgb"]

    return capture


# ---------------------------------------------------------------------------
# determinism_gate: the clean pipeline, then four ways it can be broken
# ---------------------------------------------------------------------------


def test_determinism_gate_passes_the_mock(full_style_spec: LatentSpec):
    """The mock is the known-good reference (README §4.3) -- a gate that
    can't pass it is broken, not strict."""
    backend = MockSceneBackend()
    backend.bind(full_style_spec)
    capture = _capture_factory(backend)
    base = _mid_phi(full_style_spec)
    moved = base.clone()
    moved[0, full_style_spec.index("cube.x")] += 0.2

    facts = determinism_gate(capture, base, moved)
    assert facts["order_independent_bitwise"]
    assert facts["back_to_back_bitwise"]
    assert not facts["buffers_aliased"]


class NoisyMockSceneBackend(MockSceneBackend):
    """Free-running Monte-Carlo sampling noise (README §7.1 failure (a))."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._calls = 0

    def render(self, sample_idx: int):
        self._calls += 1
        out = super().render(sample_idx)
        noisy = out["cam0"]["rgb"].clone()
        noisy[..., 0] = (noisy[..., 0].int() + self._calls) % 256
        return {"cam0": {"rgb": noisy.to(torch.uint8), "seg": out["cam0"]["seg"]}}


def test_determinism_gate_catches_sampling_noise(full_style_spec: LatentSpec):
    backend = NoisyMockSceneBackend()
    backend.bind(full_style_spec)
    capture = _capture_factory(backend)
    base, other = _mid_phi(full_style_spec), _mid_phi(full_style_spec)
    other[0, full_style_spec.index("cube.x")] += 0.2

    with pytest.raises(GateFailed) as excinfo:
        determinism_gate(capture, base, other)
    assert excinfo.value.facts["order_independent_mad"] > 0.0


class LeakyMockSceneBackend(MockSceneBackend):
    """A temporal denoiser: each frame is pulled toward the previous one --
    the x -> x' correlation README §7.1 calls the most dangerous artefact."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._previous: torch.Tensor | None = None

    def render(self, sample_idx: int):
        out = super().render(sample_idx)
        frame = out["cam0"]["rgb"].clone().float()
        if self._previous is not None:
            frame = (frame + self._previous) / 2.0
        self._previous = frame
        blended = frame.round().to(torch.uint8)
        return {"cam0": {"rgb": blended, "seg": out["cam0"]["seg"]}}


def test_determinism_gate_catches_a_temporal_leak(full_style_spec: LatentSpec):
    backend = LeakyMockSceneBackend()
    backend.bind(full_style_spec)
    capture = _capture_factory(backend)
    base, other = _mid_phi(full_style_spec), _mid_phi(full_style_spec)
    other[0, full_style_spec.index("cube.hue")] += 0.3

    with pytest.raises(GateFailed) as excinfo:
        determinism_gate(capture, base, other)
    assert excinfo.value.facts["back_to_back_mad"] > 0.0


class AliasedMockSceneBackend(MockSceneBackend):
    """Hands back one buffer, overwritten in place on every call -- the real
    Isaac camera sensor's finding (README §7.2), reproduced as a fault."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._buffer: torch.Tensor | None = None

    def render(self, sample_idx: int):
        out = super().render(sample_idx)
        fresh = out["cam0"]["rgb"]
        if self._buffer is None:
            self._buffer = fresh.clone()
        else:
            self._buffer.copy_(fresh)
        return {"cam0": {"rgb": self._buffer, "seg": out["cam0"]["seg"]}}


def test_determinism_gate_catches_an_aliased_buffer(full_style_spec: LatentSpec):
    backend = AliasedMockSceneBackend()
    backend.bind(full_style_spec)
    capture = _capture_factory(backend)
    base, other = _mid_phi(full_style_spec), _mid_phi(full_style_spec)
    other[0, full_style_spec.index("cube.hue")] += 0.3

    with pytest.raises(GateFailed) as excinfo:
        determinism_gate(capture, base, other)
    assert excinfo.value.facts["buffers_aliased"]


# ---------------------------------------------------------------------------
# read_back_gate: the frozen-write fault (IsaacSim#251)
# ---------------------------------------------------------------------------


def test_read_back_gate_passes_an_exact_write(full_style_spec: LatentSpec):
    backend = MockSceneBackend()
    backend.bind(full_style_spec)
    phi = _mid_phi(full_style_spec)
    backend.write_state(phi)
    read_back_gate(phi, backend.read_state(), atol=1e-4)


class FrozenWriteMockSceneBackend(MockSceneBackend):
    """Simulates IsaacSim#251: ``cube.x`` never actually moves off centre,
    no matter what is written -- every other role writes normally."""

    def write_state(self, phi: torch.Tensor) -> None:
        spec = self._require_bound()
        frozen = phi.clone()
        frozen[:, spec.index("cube.x")] = spec.handles[spec.index("cube.x")].center
        super().write_state(frozen)


def test_read_back_gate_catches_a_frozen_write(full_style_spec: LatentSpec):
    backend = FrozenWriteMockSceneBackend()
    backend.bind(full_style_spec)
    intended = _mid_phi(full_style_spec)
    intended[0, full_style_spec.index("cube.x")] += 0.2
    backend.write_state(intended)

    with pytest.raises(GateFailed) as excinfo:
        read_back_gate(intended, backend.read_state(), atol=1e-4)
    assert excinfo.value.facts["max_error"] > 1e-4


def test_read_back_gate_reports_the_worst_offending_dimension(full_style_spec: LatentSpec):
    backend = MockSceneBackend()
    backend.bind(full_style_spec)
    written = _mid_phi(full_style_spec)
    read_back = written.clone()
    read_back[0, 3] += 5.0  # far larger than any other planted error

    with pytest.raises(GateFailed) as excinfo:
        read_back_gate(written, read_back, atol=1e-4)
    assert excinfo.value.facts["worst_flat_index"] == 3


# ---------------------------------------------------------------------------
# style_sensitivity_gate: the disconnected-style-knob fault (README §7.5)
# ---------------------------------------------------------------------------


def test_style_sensitivity_gate_passes_when_every_knob_moves_pixels(full_style_spec: LatentSpec):
    backend = MockSceneBackend()
    backend.bind(full_style_spec)
    capture = _capture_factory(backend)
    base = _mid_phi(full_style_spec)
    perturbations = {}
    for role in ("light.intensity", "light.warmth", "cam.jitter.x"):
        phi = base.clone()
        handle = full_style_spec.handles[full_style_spec.index(role)]
        phi[0, full_style_spec.index(role)] = handle.center + 0.8 * handle.radius
        perturbations[role] = phi

    facts = style_sensitivity_gate(capture, base, perturbations, noise_floor=0.0)
    assert facts["responses"]["light.intensity"]["responsive"]


class DisconnectedStyleMockSceneBackend(MockSceneBackend):
    """The write lands and read-back is exact, but the render silently
    ignores ``light.warmth`` -- a knob the pipeline would otherwise report
    as a perfectly (and falsely) invariant style dimension."""

    def _normalized(self, role: str, spec: LatentSpec, batch_row: int) -> float:
        if role == "light.warmth":
            return 0.0
        return super()._normalized(role, spec, batch_row)


def test_style_sensitivity_gate_catches_a_disconnected_knob(full_style_spec: LatentSpec):
    """README Phase 4: 'a mock that accepts the write and renders identically
    must fail the style-sensitivity gate.' This is that mock."""
    backend = DisconnectedStyleMockSceneBackend()
    backend.bind(full_style_spec)
    capture = _capture_factory(backend)
    base = _mid_phi(full_style_spec)

    warmth_handle = full_style_spec.handles[full_style_spec.index("light.warmth")]
    moved_warmth = base.clone()
    moved_warmth[0, full_style_spec.index("light.warmth")] = (
        warmth_handle.center + 0.8 * warmth_handle.radius
    )

    intensity_handle = full_style_spec.handles[full_style_spec.index("light.intensity")]
    moved_intensity = base.clone()
    moved_intensity[0, full_style_spec.index("light.intensity")] = (
        intensity_handle.center + 0.8 * intensity_handle.radius
    )

    with pytest.raises(GateFailed) as excinfo:
        style_sensitivity_gate(
            capture,
            base,
            {"light.warmth": moved_warmth, "light.intensity": moved_intensity},
            noise_floor=0.0,
        )
    responses = excinfo.value.facts["responses"]
    assert not responses["light.warmth"]["responsive"]
    assert responses["light.intensity"]["responsive"]
