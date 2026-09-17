"""Latent specification: the registry mapping latent dimensions to physical handles.

Adding an object to the scene means appending handles -- ``n`` grows and nothing
else in the pipeline changes (README §4.1). The squash is part of the unknown
mixing map ``g``, which keeps ``z`` exactly Gaussian and unbounded (README §5.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: The only three groups a handle may belong to, and the order they must appear
#: in within a :class:`LatentSpec` (README §5.2). ``base`` is a subset of ``full``;
#: ``style`` is disjoint from both.
GROUP_ORDER: tuple[str, ...] = ("base", "full", "style")


@dataclass(frozen=True)
class Handle:
    """One latent dimension, squashed into the open interval ``center +- radius``.

    ``role`` is a backend-independent name (``"arm.j0"``, ``"cube.x"``). Each backend
    maps roles to its own targets -- the Isaac backend to a joint index resolved by
    name, the mock to a sprite parameter -- so a single spec drives both and the
    tier-1 contract suite can be parametrized over them. Franka joint names belong
    to the backend, never here.

    ``group`` is the *narrowest* of ``base``/``full``/``style`` the handle belongs
    to (README §5.2): what the downstream task needs, everything task-related a
    general version of the task would vary, and everything the task must never
    depend on. It is a property of the handle, not of a run.
    """

    role: str
    center: float
    radius: float
    group: str = "base"

    def __post_init__(self) -> None:
        if not self.radius > 0.0:
            raise ValueError(f"handle {self.role!r}: radius must be > 0, got {self.radius!r}")
        if self.group not in GROUP_ORDER:
            raise ValueError(
                f"handle {self.role!r}: group must be one of {GROUP_ORDER}, got {self.group!r}"
            )

    @classmethod
    def from_limits(
        cls, role: str, lo: float, hi: float, fraction: float = 1.0, *, group: str = "base"
    ) -> Handle:
        """Build a handle from a *measured* range, keeping ``fraction`` of its half-range.

        Radii are always a fraction of limits read back from the asset, never an
        absolute number baked into code (README §5.2).
        """
        if not hi > lo:
            raise ValueError(f"handle {role!r}: need hi > lo, got ({lo!r}, {hi!r})")
        return cls(role, 0.5 * (lo + hi), fraction * 0.5 * (hi - lo), group=group)


@dataclass(frozen=True)
class LatentSpec:
    """Ordered registry of handles; dimension ``i`` of ``z`` drives ``handles[i]``.

    Handles must appear ``base``, then ``full``, then ``style`` (README §5.2, §6.2).
    That ordering is what makes :meth:`dims` ``"base"`` a *prefix* of :meth:`dims`
    ``"full"``, which in turn is what lets latent index ``i`` mean the same physical
    thing whether a run activates ``base`` or ``full``.
    """

    handles: tuple[Handle, ...]

    def __post_init__(self) -> None:
        seen = set()
        prev_rank = -1
        for handle in self.handles:
            if handle.role in seen:
                raise ValueError(f"duplicate handle role {handle.role!r}")
            seen.add(handle.role)
            rank = GROUP_ORDER.index(handle.group)
            if rank < prev_rank:
                raise ValueError(
                    f"handles must be ordered {GROUP_ORDER}, but {handle.role!r} "
                    f"(group {handle.group!r}) follows a handle from a later group"
                )
            prev_rank = rank

    @property
    def n(self) -> int:
        return len(self.handles)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(h.role for h in self.handles)

    @property
    def group_of_dim(self) -> tuple[str, ...]:
        """Per-dimension group tag, stored with every shard (README §6.4)."""
        return tuple(h.group for h in self.handles)

    def index(self, role: str) -> int:
        """Latent dimension driving ``role``."""
        return self.roles.index(role)

    def dims(self, group: str) -> tuple[int, ...]:
        """Latent dims in ``group``.

        Cumulative for ``base``/``full`` (``dims("full")`` includes ``base`` dims
        too); disjoint for ``style``, which is never a subset of the other two.
        """
        if group not in GROUP_ORDER:
            raise ValueError(f"unknown group {group!r}, expected one of {GROUP_ORDER}")
        if group == "style":
            wanted = {"style"}
        else:
            wanted = set(GROUP_ORDER[: GROUP_ORDER.index(group) + 1]) - {"style"}
        return tuple(i for i, h in enumerate(self.handles) if h.group in wanted)

    def subset(self, group: str) -> LatentSpec:
        """A narrower spec containing only ``group``'s handles, in the same order."""
        return LatentSpec(tuple(self.handles[i] for i in self.dims(group)))

    def rho_vector(self, rho_task: float, *, rho_style: float = 0.0) -> torch.Tensor:
        """``[n]`` tensor: ``rho_task`` on base/full dims, ``rho_style`` on style dims.

        Feeds :func:`idtb.latents.ou.sample_ou_pairs`. ``rho_style`` defaults to 0
        (§5.4.1) but is a parameter, not a hardcoded zero, because the predicted
        spectrum crossing at ``rho_style = rho_task**2`` can only be tested by
        sweeping it.
        """
        return torch.tensor([rho_style if h.group == "style" else rho_task for h in self.handles])

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
