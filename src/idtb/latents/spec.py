"""Latent specification: the registry mapping latent dimensions to physical handles.

Adding an object to the scene means appending handles -- ``n`` grows and nothing
else in the pipeline changes (README §4.1). The squash is part of the unknown
mixing map ``g``, which keeps ``z`` exactly Gaussian and unbounded (README §5.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Handle:
    """One latent dimension, squashed into the open interval ``center +- radius``.

    ``role`` is a backend-independent name (``"arm.j0"``, ``"cube.x"``). Each backend
    maps roles to its own targets -- the Isaac backend to a joint index resolved by
    name, the mock to a sprite parameter -- so a single spec drives both and the
    tier-1 contract suite can be parametrized over them. Franka joint names belong
    to the backend, never here.
    """

    role: str
    center: float
    radius: float

    def __post_init__(self) -> None:
        if not self.radius > 0.0:
            raise ValueError(f"handle {self.role!r}: radius must be > 0, got {self.radius!r}")

    @classmethod
    def from_limits(cls, role: str, lo: float, hi: float, fraction: float = 1.0) -> Handle:
        """Build a handle from a *measured* range, keeping ``fraction`` of its half-range.

        Radii are always a fraction of limits read back from the asset, never an
        absolute number baked into code (README §5.2).
        """
        if not hi > lo:
            raise ValueError(f"handle {role!r}: need hi > lo, got ({lo!r}, {hi!r})")
        return cls(role, 0.5 * (lo + hi), fraction * 0.5 * (hi - lo))


@dataclass(frozen=True)
class LatentSpec:
    """Ordered registry of handles; dimension ``i`` of ``z`` drives ``handles[i]``."""

    handles: tuple[Handle, ...]

    def __post_init__(self) -> None:
        seen = set()
        for handle in self.handles:
            if handle.role in seen:
                raise ValueError(f"duplicate handle role {handle.role!r}")
            seen.add(handle.role)

    @property
    def n(self) -> int:
        return len(self.handles)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(h.role for h in self.handles)

    def index(self, role: str) -> int:
        """Latent dimension driving ``role``."""
        return self.roles.index(role)

    def squash(self, z: torch.Tensor) -> torch.Tensor:
        """phi: R^n -> physical state, ``c + r*tanh(z)``, strictly monotonic per dim.

        Injective in exact arithmetic, so ``g = render . phi`` keeps whatever
        injectivity the renderer has. See :meth:`saturated` for the float32 caveat.
        """
        center, radius = self._params(z)
        return center + radius * torch.tanh(z)

    def saturated(self, phi: torch.Tensor, *, atol: float) -> torch.Tensor:
        """Mask of handles sitting within ``atol`` of their squash bound.

        ``tanh`` is injective on paper, but in float32 the physical step per unit of
        latent collapses below any write tolerance well before overflow (|z| >~ 5):
        distinct latents then teleport the scene to the *same* state and ``g`` stops
        being injective there. Measured in physical units against the same ``atol``
        as the read-back gate -- never in latent space, where ``atanh`` inflates a
        micron of solver tolerance into an unbounded latent error.
        """
        center, radius = self._params(phi)
        return radius - (phi - center).abs() < atol

    def _params(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if ref.shape[-1] != self.n:
            raise ValueError(f"expected last dim {self.n}, got {tuple(ref.shape)}")
        kwargs = {"device": ref.device, "dtype": ref.dtype}
        return (
            torch.tensor([h.center for h in self.handles], **kwargs),
            torch.tensor([h.radius for h in self.handles], **kwargs),
        )
