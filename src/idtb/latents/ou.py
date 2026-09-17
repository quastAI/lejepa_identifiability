"""Ornstein-Uhlenbeck positive-pair sampler -- paper Eq. (1), README §6.1."""

from __future__ import annotations

import torch


def sample_ou_pairs(
    n: int,
    batch: int,
    rho: float | torch.Tensor,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``z ~ N(0, I_n)`` and ``z' = rho*z + sqrt(1-rho^2)*eta``, ``eta ~ N(0, I_n)``.

    ``rho`` is a scalar shared by every dimension, or a length-``n`` tensor built by
    ``LatentSpec.rho_vector`` -- ``rho_task`` on base/full dims, ``0.0`` on style
    dims (README §5.2). Isotropy is required only *within* the identification
    target: giving the task dimensions different rates would make that block's
    transition anisotropic, and the simultaneous (non-sequential) optimisation
    would then recover interleaved Hermite components rather than the latents
    themselves (paper App. F, README §5.4). Zeroing the style block breaks the
    same inequality in the benign direction (§5.4.1).

    Returns two ``[batch, n]`` tensors.
    """
    rho_t = torch.as_tensor(rho, dtype=dtype, device=device)
    if rho_t.ndim not in (0, 1) or (rho_t.ndim == 1 and rho_t.shape[0] != n):
        raise ValueError(
            f"rho must be a scalar or a length-{n} vector, got shape {tuple(rho_t.shape)}"
        )
    if not ((rho_t >= 0.0) & (rho_t <= 1.0)).all():
        raise ValueError(f"every rho must lie in [0, 1], got {rho!r}")
    kwargs = {"device": device, "dtype": dtype, "generator": generator}
    z = torch.randn(batch, n, **kwargs)
    eta = torch.randn(batch, n, **kwargs)
    return z, rho_t * z + (1.0 - rho_t**2) ** 0.5 * eta
