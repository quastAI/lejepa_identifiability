"""The backend seam (README §4.3): everything above this line is pure Python
and runs locally; everything below is swappable without touching the sampler
or the writer.

Pure module -- no Isaac import, ever. ``SceneBackend`` is the contract both
:class:`idtb.sim.mock.MockSceneBackend` and a future Isaac-facing backend
satisfy, and the tier-1 contract suite (``tests/contract/``) is one set of
tests run against whichever backends implement it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from idtb.latents import LatentSpec

Tensor = torch.Tensor

#: The three write mechanisms README §6.3 dispatches by -- not by latent
#: group. Each has a different failure mode (a frozen joint, an object stuck
#: at the env origin, a silently-ignored USD attribute) and therefore its own
#: read-back; a concrete backend's ``write_state`` stays explicit about which
#: path carries which role rather than collapsing to one indexed assignment.
WritePath = Literal["joint", "root_state", "attribute"]
WRITE_PATHS: tuple[WritePath, ...] = ("joint", "root_state", "attribute")


class UnsupportedRoleError(ValueError):
    """Raised by :meth:`SceneBackend.bind` for a role the backend cannot write.

    Must fire at bind time, not later as a scene silently missing a latent
    (README Phase 4). A backend that does not yet have a verified write path
    for a role -- e.g. a `style` knob no spike has confirmed -- must refuse to
    bind a spec naming it, rather than pretend to support it.
    """


@dataclass(frozen=True)
class HandleInfo:
    """What a *bound* backend can say about one role it supports."""

    role: str
    write_path: WritePath


@runtime_checkable
class SceneBackend(Protocol):
    """README §4.3. Real (Isaac, remote GPU) or mock (analytic, local) --
    everything above this line never knows which.

    Batched from the start: ``B=1`` is a valid batch, which absorbs the
    ``Camera`` vs. ``TiledCamera`` question (README §7.2 Spike 4) either way.
    """

    def bind(self, spec: LatentSpec) -> None:
        """Resolve every handle in ``spec`` to this backend's targets.

        Raises :class:`UnsupportedRoleError` immediately if ``spec`` names a
        role this backend cannot write.
        """
        ...

    def handles(self) -> Mapping[str, HandleInfo]:
        """Every role the current binding supports, and which write path
        carries it (README §6.3)."""
        ...

    def write_state(self, phi: Tensor) -> None:
        """Teleport to squashed physical state ``phi``, shape ``[B, n]``.

        No physics stepping (README §5.5's validity policy).
        """
        ...

    def read_state(self) -> Tensor:
        """``[B, n]``, for the §6.3 read-back gate."""
        ...

    def render(self, sample_idx: int) -> Mapping[str, Mapping[str, Tensor]]:
        """``{cam_id: {"rgb": [B,H,W,3] uint8, "seg": [B,H,W] int64}}``.

        Segmentation ids are the backend's own stable ids -- Isaac's own
        annotator ids are not stable across runs, which would silently turn
        the visibility column into noise (README Phase 4).
        """
        ...

    def diagnostics(self) -> Mapping[str, Tensor]:
        """Per-sample ``visibility`` and ``collision``, each ``[B]`` (§5.3, §5.5)."""
        ...

    def close(self) -> None: ...
