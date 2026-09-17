"""Tier 0: OU sampler statistics (README §10.1, first gate).

If these fail, every downstream number is meaningless.
"""

import math

import pytest
import torch

from idtb.latents import Handle, LatentSpec, sample_ou_pairs

N, BATCH = 4, 200_000


def _seeded(rho, seed=0, n=N, batch=BATCH):
    return sample_ou_pairs(n, batch, rho, generator=torch.Generator().manual_seed(seed))


def _cov(a, b):
    return (a - a.mean(0)).T @ (b - b.mean(0)) / (a.shape[0] - 1)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.95])
def test_covariance_structure(rho):
    """Cov(z) = Cov(z') = I and Cov(z, z') = rho*I."""
    z, z_next = _seeded(rho)
    eye = torch.eye(N)
    tol = 5.0 / math.sqrt(BATCH)  # ~5 sigma on a sample covariance entry
    assert torch.allclose(_cov(z, z), eye, atol=tol)
    assert torch.allclose(_cov(z_next, z_next), eye, atol=tol)
    assert torch.allclose(_cov(z, z_next), rho * eye, atol=tol)
    assert z.mean().abs() < tol and z_next.mean().abs() < tol


@pytest.mark.parametrize("rho", [0.0, 0.95])
def test_marginals_are_normal(rho):
    """Per-dimension Kolmogorov-Smirnov test against the standard normal CDF."""
    quantile = 1.63 / math.sqrt(BATCH)  # KS 99% critical value
    for sample in _seeded(rho):
        for dim in range(N):
            cdf = torch.special.ndtr(sample[:, dim].sort().values.double())
            steps = torch.arange(BATCH, dtype=torch.float64) / BATCH
            ks = max((cdf - steps).max(), (steps + 1 / BATCH - cdf).max())
            assert ks < quantile


def test_rho_one_is_the_identity():
    z, z_next = _seeded(1.0)
    assert torch.equal(z, z_next)


def test_seeding_is_reproducible():
    a, b = _seeded(0.7, seed=1, batch=16)
    c, d = _seeded(0.7, seed=1, batch=16)
    assert torch.equal(a, c) and torch.equal(b, d)
    assert not torch.equal(a, _seeded(0.7, seed=2, batch=16)[0])


def test_shape_and_dtype():
    z, z_next = sample_ou_pairs(3, 5, 0.5, dtype=torch.float64)
    assert z.shape == z_next.shape == (5, 3)
    assert z.dtype == z_next.dtype == torch.float64


@pytest.mark.parametrize("rho", [-0.1, 1.01, float("nan")])
def test_rejects_rho_outside_unit_interval(rho):
    with pytest.raises(ValueError):
        sample_ou_pairs(2, 2, rho)


# --- Vector rho: the group-wiring gate (README §5.2, §5.4.1) -------------------


def test_scalar_rho_matches_a_constant_vector_bitwise():
    scalar_z, scalar_zn = _seeded(0.95, seed=3, batch=64)
    vector_z, vector_zn = sample_ou_pairs(
        N, 64, torch.full((N,), 0.95), generator=torch.Generator().manual_seed(3)
    )
    assert torch.equal(scalar_z, vector_z)
    assert torch.equal(scalar_zn, vector_zn)


@pytest.mark.parametrize("bad_rho", [torch.tensor([0.5, 0.5]), torch.tensor([-0.1, 0.5, 0.9, 0.2])])
def test_rejects_rho_vector_of_the_wrong_length_or_out_of_range(bad_rho):
    with pytest.raises(ValueError):
        sample_ou_pairs(N, 2, bad_rho)


def test_block_cross_covariance_is_zero_exactly_on_the_style_block():
    """The only direct evidence that rho=0 reaches the style dims and no others."""
    spec = LatentSpec(
        (
            Handle("arm.j0", 0.0, 1.5, group="base"),
            Handle("cube.x", 0.4, 0.1, group="base"),
            Handle("light.intensity", 500.0, 200.0, group="style"),
        )
    )
    rho_vec = spec.rho_vector(0.95, rho_style=0.0)
    z, z_next = sample_ou_pairs(
        spec.n, BATCH, rho_vec, generator=torch.Generator().manual_seed(0)
    )
    cov = _cov(z, z_next)
    tol = 5.0 / math.sqrt(BATCH)
    task_block = spec.dims("full")  # base + full, i.e. everything but style here
    style_block = spec.dims("style")
    for i in task_block:
        assert abs(cov[i, i] - 0.95) < tol
    for i in style_block:
        assert abs(cov[i, i]) < tol
