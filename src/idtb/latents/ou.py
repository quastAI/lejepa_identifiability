"""Ornstein-Uhlenbeck positive-pair sampler -- paper Eq. (1), README §6.1."""

from __future__ import annotations

import torch


def sample_ou_pairs(
    n: int,
    batch: int,
    rho: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``z ~ N(0, I_n)`` and ``z' = rho*z + sqrt(1-rho^2)*eta``, ``eta ~ N(0, I_n)``.

    One scalar ``rho`` is shared by every dimension. Per-dimension rates would make
    the transition anisotropic, and the simultaneous (non-sequential) optimisation
    then recovers interleaved Hermite components rather than the latents themselves
    (paper App. F, README §5.4).

    Returns two ``[batch, n]`` tensors.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0, 1], got {rho!r}")
    kwargs = {"device": device, "dtype": dtype, "generator": generator}
    z = torch.randn(batch, n, **kwargs)
    eta = torch.randn(batch, n, **kwargs)
    return z, rho * z + (1.0 - rho**2) ** 0.5 * eta
