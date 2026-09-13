"""Tier 0: handle bookkeeping and the squash (README §5.1, §5.2)."""

import pytest
import torch

from idtb.latents import Handle, LatentSpec

SPEC = LatentSpec((Handle("arm.j0", 0.0, 1.5), Handle("cube.x", 0.4, 0.1)))


def test_registry_bookkeeping():
    assert SPEC.n == 2
    assert SPEC.roles == ("arm.j0", "cube.x")
    assert SPEC.index("cube.x") == 1


def test_from_limits_keeps_a_fraction_of_the_measured_half_range():
    handle = Handle.from_limits("arm.j1", lo=-1.0, hi=3.0, fraction=0.5)
    assert handle.center == 1.0
    assert handle.radius == 1.0


@pytest.mark.parametrize(
    "build",
    [
        lambda: Handle("bad", 0.0, 0.0),
        lambda: Handle.from_limits("bad", 1.0, 1.0),
        lambda: LatentSpec((Handle("dup", 0.0, 1.0), Handle("dup", 1.0, 1.0))),
    ],
)
def test_rejects_degenerate_specs(build):
    with pytest.raises(ValueError):
        build()


def test_squash_never_leaves_the_physical_range():
    """Every sampled state is physically valid by construction, for any z."""
    z = torch.linspace(-50, 50, 64).unsqueeze(1).expand(-1, SPEC.n).contiguous()
    phi = SPEC.squash(z)
    for dim, handle in enumerate(SPEC.handles):
        assert (phi[:, dim] - handle.center).abs().max() <= handle.radius


def test_squash_stays_strictly_inside_away_from_saturation():
    """Strict interiority holds until float32 tanh rounds to 1 -- see saturation test."""
    z = torch.linspace(-4, 4, 64).unsqueeze(1).expand(-1, SPEC.n).contiguous()
    phi = SPEC.squash(z)
    for dim, handle in enumerate(SPEC.handles):
        assert (phi[:, dim] - handle.center).abs().max() < handle.radius


def test_squash_is_strictly_monotonic_per_dimension():
    """Monotonicity is what keeps phi injective, hence g injective (README §5.1)."""
    z = torch.linspace(-4, 4, 500).unsqueeze(1).expand(-1, SPEC.n).contiguous()
    assert (SPEC.squash(z).diff(dim=0) > 0).all()


def test_squash_rejects_a_latent_of_the_wrong_width():
    with pytest.raises(ValueError):
        SPEC.squash(torch.zeros(3, SPEC.n + 1))


def test_saturation_flags_exactly_where_float32_injectivity_dies():
    spec = LatentSpec((Handle("arm.j0", 0.0, 1.0),))
    atol = 1e-6  # a plausible physical read-back tolerance
    z = torch.tensor([[0.0], [3.0], [9.0], [12.0]])
    phi = spec.squash(z)

    assert spec.saturated(phi, atol=atol).squeeze(1).tolist() == [False, False, True, True]
    # ...and the flagged pair really has collapsed: distinct latents, one state.
    assert phi[2] == phi[3]
    assert phi[0] != phi[1]
