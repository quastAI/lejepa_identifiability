"""Pipeline-correctness gates (README §10.1) as library code.

``generate.py`` (once the dataset driver exists) and the test suite call the
*same* functions here -- a gate that lives only in a test is not a gate a
real dataset run is protected by. Every gate raises :class:`GateFailed`
rather than returning a boolean: README §7.1 -- "if this fails, nothing else
in the project is worth running" -- and a caller cannot silently ignore a
return value it never has to check.

Pure Python/torch, no Isaac import -- these run identically against
:class:`idtb.sim.mock.MockSceneBackend` and, later, an Isaac-backed
``SceneBackend``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

Tensor = torch.Tensor
Capture = Callable[[Any], Tensor]


class GateFailed(Exception):
    """A gate ran and the answer was no. Carries the facts that showed it,
    so a caller does not have to choose between a clear message and keeping
    the diagnostic numbers behind it."""

    def __init__(self, message: str, *, facts: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.facts: dict[str, Any] = facts or {}


def _f32(t: Tensor) -> Tensor:
    """Widen before subtracting -- ``uint8`` subtraction wraps, so the largest
    possible difference (255 vs. 0) would read back as 1."""
    return t.detach().to(dtype=torch.float32)


def mean_abs_diff(a: Tensor, b: Tensor) -> float:
    return float((_f32(a) - _f32(b)).abs().mean().item())


def bitwise_equal(a: Tensor, b: Tensor) -> bool:
    a, b = a.detach(), b.detach()
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a.cpu(), b.cpu()))


def determinism_gate(
    capture: Capture, state_a: Any, state_b: Any, *, tol: float = 0.0
) -> dict[str, Any]:
    """README §7.1's acceptance test: order-independence and no back-to-back leak.

    ``capture(state)`` must return the renderer's own tensor, not a copy --
    holding a live reference to the first call is what lets this gate also
    catch an aliased sensor buffer (README §7.2), which is otherwise
    indistinguishable from a genuinely reproducible render.

    Acceptance is bitwise (``tol=0.0``) unless the caller has a specific,
    documented reason to loosen it: a temporal denoiser leak sits at a
    fraction of one grey level, under any threshold someone would plausibly
    write down, while being exactly the structure a contrastive encoder is
    trained to exploit. A threshold does not weaken this gate, it disables it.
    """
    a1_live = capture(state_a)
    a1 = a1_live.clone()
    b1 = capture(state_b).clone()
    # If capture() handed back a view onto a reused buffer, the b1 render
    # already overwrote a1_live -- A1 and a later A2 would then compare equal
    # because they are the same memory, not because the render is stable.
    aliased = not bitwise_equal(a1_live, a1)
    b2 = capture(state_b).clone()
    a2 = capture(state_a).clone()

    order_mad = mean_abs_diff(a1, a2)
    back_to_back_mad = mean_abs_diff(b1, b2)
    facts = {
        "order_independent_mad": order_mad,
        "order_independent_bitwise": bitwise_equal(a1, a2),
        "back_to_back_mad": back_to_back_mad,
        "back_to_back_bitwise": bitwise_equal(b1, b2),
        "buffers_aliased": aliased,
        "tol": tol,
    }
    if order_mad > tol or back_to_back_mad > tol:
        raise GateFailed(
            f"render is not deterministic: order_independent_mad={order_mad:.4f}, "
            f"back_to_back_mad={back_to_back_mad:.4f}, tol={tol} (README §7.1)",
            facts=facts,
        )
    if aliased:
        raise GateFailed(
            "capture() returned a live reference that a later call overwrote -- "
            "every caller must .clone() immediately, no exceptions (README §7.2)",
            facts=facts,
        )
    return facts


def read_back_gate(written: Tensor, read_back: Tensor, *, atol: float) -> dict[str, Any]:
    """README §6.3: state read back from the sim must match what was written.

    Non-negotiable. A silent write failure -- an object frozen at the env
    origin ([IsaacSim#251](https://github.com/isaac-sim/IsaacSim/issues/251)),
    a `fix_root_link`/kinematic asset, a disconnected attribute -- reads back
    as noise, not as an error, and the dataset generates normally while the
    corresponding latent is simply garbage.
    """
    if written.shape != read_back.shape:
        raise GateFailed(
            f"read-back shape {tuple(read_back.shape)} != written shape {tuple(written.shape)}"
        )
    error = (written.detach().to(torch.float64) - read_back.detach().to(torch.float64)).abs()
    max_error = float(error.max().item()) if error.numel() else 0.0
    facts = {"max_error": max_error, "atol": atol}
    if max_error > atol:
        facts["worst_flat_index"] = int(error.flatten().argmax().item())
        raise GateFailed(
            f"read-back does not match write: max error {max_error:.3g} > atol {atol:.3g} "
            f"(README §6.3)",
            facts=facts,
        )
    return facts


def style_sensitivity_gate(
    capture: Capture,
    base_state: Any,
    style_perturbations: Mapping[str, Any],
    *,
    noise_floor: float,
    min_ratio: float = 10.0,
) -> dict[str, Any]:
    """README §7.5/§10.1: every named `style` knob must move pixels far above
    the same-state noise floor when varied alone.

    A `style` knob that writes cleanly and renders deterministically but
    changes nothing visible is worse than a dead dimension: the encoder scores
    perfect invariance for free, and the headline result is then an artefact
    of an unchecked write rather than a measurement.
    """
    base = capture(base_state).clone()
    floor = max(noise_floor, 0.0)
    responses: dict[str, Any] = {}
    dead: list[str] = []
    for name, state in style_perturbations.items():
        frame = capture(state).clone()
        mad = mean_abs_diff(base, frame)
        responsive = mad > floor * min_ratio if floor > 0 else mad > 0.0
        responses[name] = {"mad_vs_base": mad, "responsive": responsive}
        if not responsive:
            dead.append(name)
    facts = {"noise_floor": floor, "min_ratio": min_ratio, "responses": responses}
    if dead:
        raise GateFailed(
            f"style knob(s) do not move pixels above the noise floor: {dead} -- a "
            "disconnected style write hands the encoder perfect invariance for free "
            "(README §7.5)",
            facts=facts,
        )
    return facts
